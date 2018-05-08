import os
import torch

import random
import argparse
import gc
import queue
import string
import time

import numpy as np
from scipy import stats
from collections import Counter
import torch.multiprocessing as mp

from model import RegexModel
import pregex as pre
from trace import Trace, RegexWrapper, PYConcept
from pinn import RobustFill
import loader
from propose import Proposal, evalProposal, getProposals, networkCache


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--fork', type=str, default=None)
parser.add_argument('--data_file', type=str, default="./data/csv.p")
parser.add_argument('--init_net', type=str, default=None)#"/om/user/lbh/text-patterns/init.pt")
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--min_examples', type=int, default=2)
parser.add_argument('--max_examples', type=int, default=4)
parser.add_argument('--max_length', type=int, default=15) #maximum length of inputs or targets
parser.add_argument('--min_iterations', type=int, default=500) #minimum number of training iterations before next concept

parser.add_argument('--n_proposals', type=int, default=10)
parser.add_argument('--n_counterproposals', type=int, default=5)
parser.add_argument('--counterexample_threshold', type=float, default=0.6)
parser.add_argument('--cell_type', type=str, default="LSTM")
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--embedding_size', type=int, default=128)

parser.add_argument('--n_tasks', type=int, default=40) #Per max_length
parser.add_argument('--skip_tasks', type=int, default=0)
parser.add_argument('--n_examples', type=int, default=500)
parser.add_argument('--initial_concepts', type=str, default='.') 

model_default_params = {'alpha':1, 'geom_p':0.5, 'pyconcept_alpha':1, 'pyconcept_d':0.1}
parser.add_argument('--alpha', type=float, default=None) #p(reference concept) proportional to #references, or to alpha if no references
parser.add_argument('--geom_p', type=float, default=None) #probability of adding another concept (geometric)
parser.add_argument('--pyconcept_alpha', type=float, default=None)
parser.add_argument('--pyconcept_d', type=float, default=None)

parser.add_argument('--helmholtz_dist', type=str, default="uniform") #During sleep, sample concepts from true weighted dist(default) or uniform

parser.add_argument('--train_first', type=int, default=0)
parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--no-network', dest='no_network', action='store_true')
parser.add_argument('--no-cuda', dest='no_cuda', action='store_true')
parser.add_argument('--debug-network', dest='debug_network', action='store_const', const=True)
parser.add_argument('--error-on-mistake', dest='error_on_mistake', action='store_const', const=True)
parser.set_defaults(debug=False, no_cuda=False, regex_primitives=False, no_network=False,debug_network=False,error_on_mistake=False)

args = parser.parse_args()
#if args.fork is None:
#	for k,v in model_default_params.items():
#		if getattr(args,k) is None: setattr(args, k, v)

default_vocabulary = list(string.printable) + \
        [pre.OPEN, pre.CLOSE, pre.String, pre.Concat, pre.Alt, pre.KleeneStar, pre.Plus, pre.Maybe]

# ----------- Network training ------------------
# Sample
def getInstance(n_examples):
	"""
	Returns a single problem instance, as input/target strings
	"""
	while True:
		r = M['trace'].model.sampleregex(M['trace'], conceptDist = args.helmholtz_dist)
		target = r.flatten()
		#inputs = ([r.sample(M['trace']) for i in range(n_examples)],)
		inputs = ([r.sample(M['trace']) for i in range(n_examples)],)
		if len(target)<args.max_length and all(len(x)<args.max_length for x in inputs[0]): break
	return {'inputs':inputs, 'target':target}

def getBatch(batch_size):
	"""
	Create a batch of problem instances, as tensors
	"""
	n_examples = random.randint(args.min_examples, args.max_examples)
	instances = [getInstance(n_examples) for i in range(batch_size)]
	inputs = [x['inputs'] for x in instances]
	target = [x['target'] for x in instances]
	return inputs, target

# SGD
def refreshVocabulary():
	#Make sure network/optimiser are updated to current vocabulary.
	M['net'] = M['net'].with_target_vocabulary(default_vocabulary + M['trace'].baseConcepts)


def networkStep():
	inputs, target = getBatch(args.batch_size)
	network_score = M['net'].optimiser_step(inputs, target)

	M['state']['network_losses'].append(-network_score)
	M['state']['iteration'] += 1

	if M['state']['iteration']%10==0 and args.debug_network: print(inputs[0], target[0], net.sample(inputs)[0])

	networkCache.clear()
	return network_score

def train(toConvergence=False, iterations=None, saveEvery=500):
	refreshVocabulary()
	from_iteration = M['state']['task_iterations'][-1] if M['state']['task_iterations'] else 0
	while True:
		if toConvergence:
			networkStep()
			
			if M['state']['iteration']%10==0:
				window_size = args.min_iterations
				window = M['state']['network_losses'][max(from_iteration, len(M['state']['network_losses']) - window_size):]
				regress = stats.linregress(range(len(window)), window)
				
				print("Iteration %d" % M['state']['iteration'], "| Network loss: %2.2f" % M['state']['network_losses'][-1], "| Slope: %4.4f" % regress.slope)
				if len(M['state']['network_losses']) >= from_iteration + window_size and regress.slope>-0.0001:
					break #Break when converged
		else:
			networkStep()
			if M['state']['iteration']%10==0:
				print("Iteration %d" % M['state']['iteration'], "| Network loss: %2.2f" % M['state']['network_losses'][-1])
			if len(M['state']['network_losses']) >= from_iteration + iterations:
				break

		if not args.debug and len(M['state']['network_losses']) % saveEvery == 0:
			loader.save(M)


# ----------- Solve a task ------------------
def onCounterexamples(queueProposal, proposal, counterexamples, p_valid, kinkscore=None, nEffectiveExamples=None):
	if p_valid>0.5 and proposal.depth==0:
		if kinkscore is None or kinkscore < args.counterexample_threshold:
			counterexamples_unique = list(set(counterexamples))

			#Retry by including counterexamples in support set
			sampled_counterexamples = np.random.choice(counterexamples_unique, size=min(len(counterexamples_unique), 5), replace=False)
			counterexample_proposals = getProposals(M['net'] if not args.no_network else None, proposal.trace, tuple(proposal.examples) + tuple(sampled_counterexamples),
					depth=proposal.depth+1, nProposals=args.n_counterproposals, nEffectiveExamples=nEffectiveExamples)
			for counterexample_proposal in counterexample_proposals:
				print("(kink score %2.2f)" % (kinkscore or 0), "ADDING:", counterexample_proposal.concept.str(counterexample_proposal.trace), "for counterexamples:", sampled_counterexamples, "on", proposal.concept.str(proposal.trace), flush=True)
				queueProposal(counterexample_proposal)
			
			#Deal with counter examples separately (with Alt)
			sampled_counterexamples = np.random.choice(counterexamples, size=min(len(counterexamples), 5), replace=False)
			counterexample_proposals = getProposals(M['net'] if not args.no_network else None, proposal.trace, sampled_counterexamples,
					depth=proposal.depth+1, nProposals=args.n_counterproposals, nEffectiveExamples=nEffectiveExamples)
			for counterexample_proposal in counterexample_proposals: 
				trace, concept = counterexample_proposal.trace.addregex(pre.Alt(
					[RegexWrapper(proposal.concept), RegexWrapper(counterexample_proposal.concept)], 
					ps = [p_valid, 1-p_valid]))
				new_proposal = Proposal(proposal.depth+1, tuple(proposal.examples) + tuple(sampled_counterexamples), trace, concept, None, None, None)
				print("(kink score %2.2f)" % (kinkscore or 0), "ADDING:", new_proposal.concept.str(new_proposal.trace), "for counterexamples:", sampled_counterexamples, "on", proposal.concept.str(proposal.trace), flush=True)
				queueProposal(new_proposal)
		else:
			print("(kink score %2.2f)" % kinkscore, "for", counterexamples[:5], "on", proposal.concept.str(proposal.trace))


def cpu_worker(worker_idx, init_trace, q_proposals, q_counterexamples, q_solutions, l_active, task_idx, task):
	solutions = []
	nEvaluated = 0

	while any(l_active) or not q_counterexamples.empty():
		try:
			proposal = q_proposals.get(timeout=1)
		except queue.Empty:
			l_active[worker_idx] = False
			continue

		l_active[worker_idx] = True
		start_time=time.time()
		solution = evalProposal(proposal, task, onCounterexamples=lambda *args: q_counterexamples.put(args), doPrint=False, task_idx=task_idx)
		took = time.time()-start_time

		nEvaluated += 1
		if solution.valid:
			solutions.append(solution)
			print("(Worker %d, %2.2fs)"%(worker_idx, took), "Score: %3.3f"%(solution.final_trace.score - init_trace.score), "(prior %3.3f + likelihood %3.3f):"%(solution.trace.score - init_trace.score, solution.final_trace.score - solution.trace.score), proposal.concept.str(proposal.trace), flush=True)
		else:
			print("(Worker %d, %2.2fs)"%(worker_idx, took), "Failed:", proposal.concept.str(proposal.trace), flush=True)
		
	q_solutions.put(
		{"nEvaluated": nEvaluated,
		 "nSolutions": len(solutions),
		 "best": max(solutions, key=lambda evaluatedProposal: evaluatedProposal.final_trace.score) if solutions else None
		})

def addTask(task_idx):
	print("\n" + "*"*40 + "\nAdding task %d (n=%d)" % (task_idx, len(data[task_idx])))
	print("Task: " + ", ".join(list(set(data[task_idx]))))

	example_counter = Counter(data[task_idx])
	# q_proposals = mp.Queue()
	
	manager = mp.Manager()
	q_proposals = manager.Queue()

	proposal_strings_sofar = [] #TODO: this better. Want to avoid duplicate proposals. For now, just using string representation to check...

	def queueProposal(proposal): #add to evaluation queue
		proposal = proposal.strip() #Remove any evaluation data
		proposal_string = proposal.concept.str(proposal.trace, depth=-1) 
		if proposal_string not in proposal_strings_sofar:
			q_proposals.put(proposal)
			proposal_strings_sofar.append(proposal_string)

	n_workers = max(1, cpus-1)
	q_counterexamples = manager.Queue()
	q_solutions = manager.Queue()
	l_active = manager.list([True for _ in range(n_workers)])

	def launchWorkers():
		for worker_idx in range(n_workers):
			mp.Process(target=cpu_worker, args=(worker_idx, M['trace'], q_proposals, q_counterexamples, q_solutions, l_active, task_idx, data[task_idx])).start()

	for i in range(10 if not args.debug else 3):
		num_examples = random.randint(args.min_examples, args.max_examples)
		examples = list(np.random.choice(
			list(example_counter.keys()),
			size=min(num_examples, len(example_counter)),
			p=np.array(list(example_counter.values()))/sum(example_counter.values()),
			replace=True))
		pre_trace = M['trace']
		new_proposals = getProposals(M['net'] if not args.no_network else None, pre_trace, examples, nProposals=args.n_proposals)
		for proposal in new_proposals:	queueProposal(proposal)
		if i==0: launchWorkers()

	while any(l_active) or not q_counterexamples.empty():
		try:
			counterexample_args = q_counterexamples.get(timeout=0.1)
			onCounterexamples(queueProposal, *counterexample_args, nEffectiveExamples=len(data[task_idx]))
		except queue.Empty:
			pass
			#if not args.no_network: networkStep()

	solutions = []
	nSolutions = 0
	nEvaluated = 0
	for worker_idx in range(n_workers):
		x = q_solutions.get()
		nEvaluated += x['nEvaluated']
		nSolutions += x['nSolutions']
		if x['best'] is not None: solutions.append(x['best'])
	
	print("Evaluated", nEvaluated, "proposals", "(%d solutions)" % nSolutions)
	accepted = max(solutions, key=lambda evaluatedProposal: evaluatedProposal.final_trace.score)
	print("Accepted proposal: " + accepted.concept.str(accepted.trace) + "\nScore:" + str(accepted.final_trace.score - M['trace'].score) + "\n")
	M['trace'] = accepted.final_trace
	M['task_observations'][task_idx] = accepted.observations
	M['task_concepts'][task_idx] = accepted.concept
	#refreshVocabulary()
	M['state']['task_iterations'].append(M['state']['iteration'])

def checkForMistakes():
	upper_concept = next((c for c in M['trace'].baseConcepts if type(c) is PYConcept and all(x in M['trace'].getState(c).value_tables.keys() for x in 'ABCDEFG')), None)
	digit_concept = next((c for c in M['trace'].baseConcepts if type(c) is PYConcept and all(x in M['trace'].getState(c).value_tables.keys() for x in '1234567890')), None)
	if upper_concept is not None:
		assert not any(x not in string.ascii_uppercase for x in M['trace'].getState(upper_concept).value_tables.keys()), "Uppercase concept failed"
	if digit_concept is not None:
		assert not any(x not in string.digits for x in M['trace'].getState(digit_concept).value_tables.keys()), "Digits concept failed"

# -----------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
	mp.set_start_method('spawn')

	# Compute
	if "SLURM_CPUS_PER_TASK" in os.environ: cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
	else: cpus = 1
	print("Running on %d CPUs" % cpus)

	# Files to save:
	save_to = "results/"
	if not os.path.exists(save_to): os.makedirs(save_to)
	modelfile = save_to + "model.pt"
	use_cuda = torch.cuda.is_available() and not args.no_cuda

	# ------------- Load Model & Data --------------
	# Data
	data, group_idxs, test_data = loader.loadData(args.data_file, args.n_examples, args.n_tasks, args.max_length)

	# Model
	M = None
	if not args.debug:
		try:
			M = loader.load(modelfile)
			print("Loaded model ", modelfile)
			M['args'] = args

		except FileNotFoundError:
			if args.fork is not None:
				M = loader.load(args.fork, use_cuda)
				M['args'] = args
				print("Forked model", args.fork)
	
	if M is not None:
		for param in model_default_params:
			val = getattr(args, param)
			if val is not None:
				setattr(M['trace'].model, param, val)
				print("set model." + str(param) + "=" + str(val))
	else:
		M = {}
		M['state'] = {'iteration':0, 'current_task':0, 'network_losses':[], 'task_iterations':[]}
		
		if args.init_net is None: 
			M['net'] = net = RobustFill(input_vocabularies=[string.printable], target_vocabulary=default_vocabulary,
										 hidden_size=args.hidden_size, embedding_size=args.embedding_size, cell_type=args.cell_type)
			print("Created new network")
		else:
			_M = loader.load(args.init_net)
			M['net'] = net = _M['net'] 
			M['state']['network_losses'] = _M['state']['network_losses']
			M['state']['iteration'] = _M['state']['iteration']
			assert(net.hidden_size==args.hidden_size and net.embedding_size==args.embedding_size and net.cell_type==args.cell_type)
			print("Loaded existing network")
		
		M['args'] = args
		M['task_observations'] = [[] for _ in range(len(data))]
		M['task_concepts'] = [[] for _ in range(len(data))]
		M['trace'] = Trace(model=RegexModel(
			alpha=args.alpha if args.alpha is not None else model_default_params['alpha'],
			geom_p=args.geom_p if args.geom_p is not None else model_default_params['geom_p'],
			pyconcept_alpha=args.pyconcept_alpha if args.pyconcept_alpha is not None else model_default_params['pyconcept_alpha'],
			pyconcept_d=args.pyconcept_d if args.pyconcept_d is not None else model_default_params['pyconcept_d']))

		for (s, c) in [(".", pre.dot), ("d", pre.d), ("s", pre.s), ("w", pre.w), ("l", pre.l), ("u", pre.u)]:
			if s in args.initial_concepts: M['trace'], init_concept = M['trace'].initregex(c)

		print("Created new model")
	
	M['data_file'] = args.data_file
	M['save_to'] = save_to

	if use_cuda: M['net'].cuda()

	print("\nTraining...")
	#refreshVocabulary()

	def save(saveNet=False):
		print("Saving...")
		loader.saveCheckpoint(M, saveNet)
		loader.saveRender(M)
		loader.save(M)
		print("Saved.")

	if args.train_first > 0: train(iterations=args.train_first)

	for i in range(M['state']['current_task'], len(data)):
		if (i==0 or i in group_idxs) and not args.no_network and not (i==0 and args.init_net is not None):
			train(toConvergence=True)
			gc.collect()
			if not args.debug: save(saveNet=True)

		print("\n" + str(len(M['trace'].baseConcepts)) + " concepts:", ", ".join(c.str(M['trace'], depth=1) for c in M['trace'].baseConcepts))
		addTask(M['state']['current_task'])
		if args.error_on_mistake: checkForMistakes()	
		M['state']['current_task'] += 1
		gc.collect()
		if not args.debug: save()
