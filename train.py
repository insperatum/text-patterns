import os
import torch

import random
import argparse
import gc
import queue
import string
import time

from collections import namedtuple

import numpy as np
from scipy import stats
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
parser.add_argument('--data_file', type=str, default="./data/csv_900.p")
parser.add_argument('--net', type=str, default=None)
parser.add_argument('--init_net', type=str, default="/om/user/lbh/text-patterns/init.pt")
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--min_train_examples', type=int, default=1)
parser.add_argument('--min_examples', type=int, default=2)
parser.add_argument('--max_examples', type=int, default=4)
parser.add_argument('--max_length', type=int, default=15) #maximum length of inputs or targets
parser.add_argument('--iterations', type=int, default=2500) #number of network training iterations before each curriculum stage
parser.add_argument('--timeout', type=int, default=5) #minutes per task
parser.add_argument('--n_proposals', type=int, default=100)
parser.add_argument('--n_counterproposals', type=int, default=5)
parser.add_argument('--counterexample_depth', type=int, default=2)
parser.add_argument('--counterexample_threshold', type=float, default=0.8)
parser.add_argument('--counterexample_threshold2', type=float, default=0.4)
parser.add_argument('--cell_type', type=str, default="LSTM")
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--embedding_size', type=int, default=128)

parser.add_argument('--n_tasks', type=int, default=40) #Per max_length
parser.add_argument('--skip_tasks', type=int, default=0)
parser.add_argument('--n_examples', type=int, default=100)
parser.add_argument('--initial_concepts', type=str, default='.') 

model_default_params = {'alpha':1, 'geom_p':0.5, 'pyconcept_alpha':1, 'pyconcept_d':0.1, 'pyconcept_threshold':0.0001}
parser.add_argument('--alpha', type=float, default=None) #p(reference concept) proportional to #references+alpha
parser.add_argument('--geom_p', type=float, default=None) #probability of adding another concept (geometric)
parser.add_argument('--pyconcept_alpha', type=float, default=None)
parser.add_argument('--pyconcept_d', type=float, default=None)
parser.add_argument('--pyconcept_threshold', type=float, default=None)
parser.add_argument('--helmholtz_dist', type=str, default="default") #During sleep, sample concepts from true weighted dist(default) or uniform

parser.add_argument('--train_first', type=int, default=0)
parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--no-network', dest='no_network', action='store_true')
parser.add_argument('--no-cuda', dest='no_cuda', action='store_true')
parser.add_argument('--debug-network', dest='debug_network', action='store_const', const=True)
parser.add_argument('--error-on-mistake', dest='error_on_mistake', action='store_const', const=True)
parser.add_argument('--no-network-on-alt', dest='no_network_on_alt', action='store_const', const=True)
parser.add_argument('--no-depth2-network', dest='no_depth2_network', action='store_const', const=True)
parser.add_argument('--no-alt-on-counterexample', dest='no_alt_on_counterexample', action='store_const', const=True)
parser.add_argument('--no-save', dest='no_save', action='store_const', const=True)
parser.set_defaults(debug=False, no_cuda=False, regex_primitives=False, no_network=False,debug_network=False,error_on_mistake=False,no_network_on_alt=False,no_depth2_network=False,no_alt_on_counterexample=False,no_save=False)

args = parser.parse_args()
if __name__=="__main__":
	for k,v in vars(args).items():
		if v is None and k in model_default_params:
			print(k, "=", model_default_params[k], "(default)")
		else:
			print(k, "=", v)
	print()
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
	n_examples = random.randint(args.min_train_examples, args.max_examples)
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
				window_size = 1000
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

		if not args.no_save and len(M['state']['network_losses']) % saveEvery == 0:
			loader.save(M)


# ----------- Solve a task ------------------
#q_main: (priority, type, value)
#priority: (priority-type, guess at (negative) final score, rand)
#priority-type: 0=guaranteed solution, known score,
#               1=guaranteed solution, unknown score (either from found partial solution, or from taking counterexamples to a solution that definitely works)
# 				2=unknown
#type: "solution", "partialSolution", "counterexample"
#value: ...
QueueItem = namedtuple("QueueItem", ["priority", "type", "value"])

def queueSolution(q_main, solution):
	priority = (0, -solution.final_trace.score, random.random())
	q_main.put(QueueItem(priority, "solution", solution))

def queuePartialSolution(q_main, partialSolution):
	if partialSolution.altWith.altWith is None:
		priority = (1, float("inf"), random.random())
	else:
		priority = (2, float("inf"), random.random())
	q_main.put(QueueItem(priority, "partialSolution", partialSolution))

def queueCounterexamples(q_main, counterexample_args):
	(proposal, counterexamples, p_valid, kinkscore) = counterexample_args
	if kinkscore is not None:
		if proposal.altWith is None:
			priority = (1, -proposal.final_trace.score, random.random())
		else:
			priority = (2, float("inf"), random.random())
	else:
		priority = (2, float("inf"), random.random())
	q_main.put(QueueItem(priority, "counterexamples", counterexample_args))

def queueInitial(q_main):
	priority = (2, 0, random.random())
	q_main.put(QueueItem(priority, "initial", None))

def onCounterexamples(queueProposal, proposal, counterexamples, p_valid, kinkscore=None):
	if p_valid>0.5 and proposal.depth<args.counterexample_depth:
		counterexample_threshold = args.counterexample_threshold if proposal.depth==0 or args.counterexample_threshold2 is None else args.counterexample_threshold2
		
		if kinkscore is None or kinkscore < counterexample_threshold:
			#Retry by including counterexamples in support set
			unique_counterexamples = list(set(counterexamples))
			sampled_counterexamples = np.random.choice(unique_counterexamples, size=min(len(unique_counterexamples), 3), replace=False)
			counterexample_proposals = getProposals(M['net'] if not (args.no_network or (proposal.depth+1==2 and args.no_depth2_network)) else None, proposal.init_trace, proposal.target_examples,
					net_examples=(tuple(proposal.net_examples) + tuple(sampled_counterexamples))[-5:], depth=proposal.depth+1, nProposals=args.n_counterproposals, altWith=proposal.altWith)

			for counterexample_proposal in counterexample_proposals:
				print("(depth %d kink %2.2f)" % (counterexample_proposal.depth, kinkscore or 0),
					"adding joint", counterexample_proposal.concept.str(counterexample_proposal.trace),
					"for counterexamples:", sampled_counterexamples, "on", proposal.concept.str(proposal.trace), 
					flush=True)
				queueProposal(counterexample_proposal)
			
			if not args.no_alt_on_counterexample:
				#Deal with counter examples separately (with Alt)	
				sampled_counterexamples = np.random.choice(counterexamples, size=min(len(counterexamples), 4), replace=False)
				unique_counterexamples = list(set(counterexamples))
				for counterexample_proposal in getProposals(M['net'] if not (args.no_network or args.no_network_on_alt or (proposal.depth+1==2 and args.no_depth2_network)) else None, proposal.trace, counterexamples,
					net_examples=sampled_counterexamples, depth=proposal.depth+1, nProposals=args.n_counterproposals, altWith=proposal):
					queueProposal(counterexample_proposal)
					print("(depth %d kink %2.2f)" % (counterexample_proposal.depth, kinkscore or 0),
						"adding exception", counterexample_proposal.concept.str(counterexample_proposal.trace),
						"for counterexamples:", sampled_counterexamples, "on", proposal.concept.str(proposal.trace), 
						flush=True)
		else:
			print("(depth %d kink %2.2f)" % (proposal.depth, kinkscore), "for", counterexamples[:5], "on", proposal.concept.str(proposal.trace), flush=True)

def onPartialSolution(partialSolution, queueProposal, getRelated):
	#Add (X|Y)
	p = len(partialSolution.target_examples) / len(partialSolution.altWith.target_examples)
	trace, concept = partialSolution.trace.addregex(pre.Alt(
		[RegexWrapper(partialSolution.altWith.concept), RegexWrapper(partialSolution.concept)], 
		ps = [1-p, p]))
	new_proposal = Proposal(partialSolution.depth, partialSolution.altWith.net_examples + partialSolution.net_examples,
			partialSolution.altWith.target_examples, partialSolution.init_trace, trace, concept, (), partialSolution.altWith.altWith, None, None, None)
	queueProposal(new_proposal, highPriority=partialSolution.altWith.altWith is not None)

	#ADD PY(X|Y)
	trace, concept = trace.addPY(concept)
	new_proposal = Proposal(partialSolution.depth, partialSolution.altWith.net_examples + partialSolution.net_examples,
			partialSolution.altWith.target_examples, partialSolution.init_trace, trace, concept, (), partialSolution.altWith.altWith, None, None, None)
	queueProposal(new_proposal, highPriority=partialSolution.altWith.altWith is not None)

	#Add related proposals for X 
	#TODO: it's a bit silly to have to do this, we know it's going to fail, but then it'll generate the right traces for possibly better partialsolutions
	for relatedProposal in getRelated(partialSolution.altWith):
		queueProposal(relatedProposal)
	
	

def cpu_worker(worker_idx, init_trace, l_proposals, l_partialProposals, q_main, l_active, l_running, task_idx, task):
	nEvaluated = 0

	while l_running[0]:
		try:
			proposal = l_proposals.pop(0)
			assert(proposal.altWith is None)
		except IndexError:
			try:
				proposal = l_partialProposals.pop(0)
				assert(proposal.altWith is not None)
			except IndexError:
				l_active[worker_idx] = False
				continue

		l_active[worker_idx] = True
		start_time=time.time()
		def handle_counterexamples(*args):
			proposal, counterexamples, p_valid, kinkscore = args
			print(proposal.concept.str(proposal.trace), "counterexamples:", list(set(counterexamples))[:5], flush=True)
			queueCounterexamples(q_main, args)
		solution = evalProposal(proposal, onCounterexamples=handle_counterexamples, doPrint=False, task_idx=task_idx)
		took = time.time()-start_time

		if proposal.altWith is None:
			nEvaluated += 1
			if solution.valid:
				#solutions.append(solution)
				#q_solutions.put(solution)
				queueSolution(q_main, solution)
				print("(Worker %d, %2.2fs)"%(worker_idx, took), "Score: %3.3f"%(solution.final_trace.score - init_trace.score), "(prior %3.3f + likelihood %3.3f):"%(solution.trace.score - init_trace.score, solution.final_trace.score - solution.trace.score), proposal.concept.str(proposal.trace), flush=True)
				if tuple(solution.target_examples) != tuple(task):
					print("INCORRECT TARGET EXAMPLES??", solution.concept.str(solution.trace), solution.target_examples)
					raise Exception()

			else:
				print("(Worker %d, %2.2fs)"%(worker_idx, took), "Failed:", proposal.concept.str(proposal.trace), flush=True)
		else:
			if solution.valid:
				queuePartialSolution(q_main, solution)
				print("(Worker %d, %2.2fs)"%(worker_idx, took), "Got partial solution", proposal.concept.str(proposal.trace), flush=True)
			else:
				print("(Worker %d, %2.2fs)"%(worker_idx, took), "Failed partial", proposal.concept.str(proposal.trace), flush=True)
		
	#q_solutions.put(
	#	{"nEvaluated": nEvaluated,
	#	 "nSolutions": len(solutions),
	#	 "best": max(solutions, key=lambda evaluatedProposal: evaluatedProposal.final_trace.score) if solutions else None
	#	})

def addTask(task_idx):
	addTaskStart = time.time()
	print("\n" + "*"*40 + "\nAdding task %d (n=%d)" % (task_idx, len(data[task_idx])))
	print("Task: " + ", ".join(list(set(data[task_idx]))))

	# q_proposals = mp.Queue()
	
	manager = mp.Manager()
	#q_proposals = manager.Queue()
	#q_partialProposals = manager.Queue()
	l_proposals = manager.list()
	l_partialProposals = manager.list()
	def getProposalID(proposal): #To avoid duplicate proposals
		return (proposal.concept.str(proposal.trace, depth=-1),
				getProposalID(proposal.altWith) if proposal.altWith is not None else None)
	proposalIDs_so_far = []
	relatedProposalsDict = {}
	def queueProposal(proposal, highPriority=False): #add to evaluation queue
		proposal = proposal.strip()
		proposalID = getProposalID(proposal)
		relatedProposalsDict[proposalID] = proposal.related
		proposal = proposal._replace(related=())
		if proposalID not in proposalIDs_so_far:
			proposalIDs_so_far.append(proposalID)
			if proposal.altWith is None:
				if highPriority:
					l_proposals.insert(0, proposal)
				else:
					l_proposals.append(proposal)
			else:
				if highPriority:
					l_partialProposals.insert(0, proposal)
				else:
					l_partialProposals.append(proposal)

	def getRelated(proposal):
		return relatedProposalsDict[getProposalID(proposal)]

	def addRelated(solution):
		related = getRelated(solution)
		if len(related)>0:
			print("Add related proposals", solution.concept.str(solution.trace), "---->", ", ".join(p.concept.str(p.trace) for p in related))
		for p in related:
			queueProposal(p)

	n_workers = max(1, cpus-1)
	q_main = manager.Queue()
	pq_main = queue.PriorityQueue()
	solutions = []
	l_active = manager.list([True for _ in range(n_workers)])
	l_running = manager.list([True])
	workers = []
	def launchWorkers():
		for worker_idx in range(n_workers):
			worker = mp.Process(target=cpu_worker, args=(worker_idx, M['trace'], l_proposals, l_partialProposals, q_main, l_active, l_running, task_idx, data[task_idx]))
			workers.append(worker)
			worker.start()

	init_proposals = getProposals(M['net'] if not args.no_network else None, M['trace'], data[task_idx], nProposals=args.n_proposals, subsampleSize=(args.min_examples,args.max_examples))
	for i in range(args.n_proposals-1): queueInitial(q_main)
	queueProposal(next(init_proposals))
	launchWorkers()


	startTime = time.time()
	#partialSolutionsByAltWith = {}
	bestPartialSolutionByAltWith = {}
	while any(l_active) or len(l_proposals)>0 or len(l_partialProposals)>0 or not q_main.empty() or not pq_main.empty():# or len(partialSolutionsByAltWith)>0:
		if time.time() - startTime > args.timeout * 60:
			print("Timeout!")
			break

		try:
			pq_main.put(q_main.get(timeout=0.1))
			continue
		except queue.Empty:
			try:
				queue_item = pq_main.get_nowait()
			except queue.Empty:
				continue
	

		assert(queue_item.type in ["solution", "partialSolution", "counterexamples", "initial"])
		#Initial Proposal
		if queue_item.type == "initial":
			try:
				queueProposal(next(init_proposals))
			except StopIteration:
				pass

		#Solutions
		if queue_item.type == "solution":
			solution = queue_item.value
			solutions.append(solution)
			addRelated(solution)

		#PartialSolutions
		if queue_item.type == "partialSolution":
			partialSolution = queue_item.value 
			#if getProposalID(partialSolution.altWith) not in partialSolutionsByAltWith: partialSolutionsByAltWith[getProposalID(partialSolution.altWith)]=[]
			#partialSolutionsByAltWith[getProposalID(partialSolution.altWith)].append(partialSolution)
			altWithID = getProposalID(partialSolution.altWith)
			if altWithID not in bestPartialSolutionByAltWith or partialSolution.final_trace.score > bestPartialSolutionByAltWith[altWithID].final_trace.score:
				bestPartialSolutionByAltWith[altWithID] = partialSolution
				addRelated(partialSolution)
				onPartialSolution(partialSolution, queueProposal, getRelated)
		#if len(partialSolutionsByAltWith)>0 and not any(l_active):
		#	#print("Reading partial solutions for:", list(partialSolutionsByAltWith.keys()))
		#	#for ps in partialSolutionsByAltWith.values():
		#	remove = []
		#	for (altWithID, ps) in partialSolutionsByAltWith.items():
		#		if not any (getProposalID(x.altWith)==altWithID for x in l_partialProposals):
		#			partialAccepted = max(ps, key=lambda evaluatedProposal: evaluatedProposal.final_trace.score)
		#			print("Reading partial solutions(", altWithID, "):", list(x.concept.str(x.trace) + "=" + str(x.final_trace.score) for x in ps), "\nBest was:", getProposalID(partialAccepted))
		#			addRelated(partialAccepted)
		#			onPartialSolution(partialAccepted, queueProposal, getRelated)
		#			remove.append(altWithID)
		#	for altWithID in remove: del partialSolutionsByAltWith[altWithID]

		#Counterexamples
		if queue_item.type == "counterexamples":
			counterexample_args = queue_item.value 
			onCounterexamples(queueProposal, *counterexample_args)

	l_running[0] = False

	for w in workers:
		if w.is_alive() :w.terminate()

	print("Found %d solutions" % len(solutions))
	if len(solutions)==0:
		print("No solutions??")
		addTask(task_idx)
	else:
		accepted = max(solutions, key=lambda evaluatedProposal: evaluatedProposal.final_trace.score)
		print("Accepted proposal: " + accepted.concept.str(accepted.trace) + "\nScore:" + str(accepted.final_trace.score - M['trace'].score) + "\n")
		print("\nBest candidates were:")
		for solution in sorted(solutions, key=lambda evaluatedProposal: evaluatedProposal.final_trace.score, reverse=True)[:10]:
			print(solution.concept.str(solution.trace, depth=3) + 
				"Score: %3.3f"%(solution.final_trace.score - M['trace'].score),
				"(prior %3.3f + likelihood %3.3f):"%(
					solution.trace.score - M['trace'].score,
					solution.final_trace.score - solution.trace.score))
		print("addTask took", time.time()-addTaskStart, "seconds")
		M['trace'] = accepted.final_trace
		M['task_observations'][task_idx] = accepted.observations
		M['task_concepts'][task_idx] = accepted.concept
		#refreshVocabulary()
		M['state']['task_iterations'].append(M['state']['iteration'])

def checkForMistakes():
	upper_concept = next((c for c in M['trace'].baseConcepts if type(c) is PYConcept and all(x in M['trace'].getState(c).value_tables.keys() for x in 'ABCDEFG')), None)
	digit_concept = next((c for c in M['trace'].baseConcepts if type(c) is PYConcept and all(x in M['trace'].getState(c).value_tables.keys() for x in '1234567890')), None)
	if upper_concept is not None:
		if any(x not in string.ascii_uppercase for x in M['trace'].getState(upper_concept).value_tables.keys()):
			if args.error_on_mistake:
				raise Exception("Uppercase concept failed")
			else:
				if 'mistake_on_task' not in M:
					print("Uppercase concept failed")
					M['mistake_on_task']=M['state']['current_task']
	if digit_concept is not None:
		if any(x not in string.digits for x in M['trace'].getState(digit_concept).value_tables.keys()):
			if args.error_on_mistake:
				raise Exception("Digits concept failed")
			else:
				if 'mistake_on_task' not in M:
					print("Digits concept failed")
					M['mistake_on_task']=M['state']['current_task']

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

	def loadNet(path):
		_M = loader.load(path)
		M['net'] = net = _M['net'] 
		M['state']['network_losses'] = _M['state']['network_losses']
		M['state']['iteration'] = _M['state']['iteration']
		assert(net.hidden_size==args.hidden_size and net.embedding_size==args.embedding_size and net.cell_type==args.cell_type)
		print("Loaded network:", path)

	if M is not None:
		for param in model_default_params:
			val = getattr(args, param)
			if val is not None:
				setattr(M['trace'].model, param, val)
				print("set model." + str(param) + "=" + str(val))
	else:
		M = {}
		M['state'] = {'iteration':0, 'current_task':0, 'network_losses':[], 'task_iterations':[]}
	
		if not args.no_network:
			if args.init_net is not None:
				loadNet(args.init_net)	
			elif args.net is None: 
				M['net'] = net = RobustFill(input_vocabularies=[string.printable], target_vocabulary=default_vocabulary,
											 hidden_size=args.hidden_size, embedding_size=args.embedding_size, cell_type=args.cell_type)
				print("Created new network")
			#else:
		else:
			M['net']=None
		
		M['args'] = args
		M['task_observations'] = [[] for _ in range(len(data))]
		M['task_concepts'] = [[] for _ in range(len(data))]
		M['trace'] = Trace(model=RegexModel(
			alpha=args.alpha if args.alpha is not None else model_default_params['alpha'],
			geom_p=args.geom_p if args.geom_p is not None else model_default_params['geom_p'],
			pyconcept_alpha=args.pyconcept_alpha if args.pyconcept_alpha is not None else model_default_params['pyconcept_alpha'],
			pyconcept_d=args.pyconcept_d if args.pyconcept_d is not None else model_default_params['pyconcept_d'],
			pyconcept_threshold=args.pyconcept_threshold if args.pyconcept_threshold is not None else model_default_params['pyconcept_threshold']))

		for (s, c) in [(".", pre.dot), ("d", pre.d), ("s", pre.s), ("w", pre.w), ("l", pre.l), ("u", pre.u)]:
			if s in args.initial_concepts: M['trace'], init_concept = M['trace'].initregex(c)

		print("Created new model")

	if args.net is not None:
		loadNet(args.net)
	
	M['data_file'] = args.data_file
	M['save_to'] = save_to

	if use_cuda and M['net'] is not None: M['net'].cuda()

	print("\nTraining...")
	#refreshVocabulary()

	def save(saveNet=False):
		print("Saving...")
		loader.saveCheckpoint(M, saveNet)
		loader.saveRender(M)
		loader.save(M)
		print("Saved.")

	if args.train_first > 0: train(iterations=args.train_first)

	print("Starting from task", M['state']['current_task'])
	for i in range(M['state']['current_task'], len(data)):
		if (i==0 or i in group_idxs) and not args.no_network and not (i==0 and args.net is not None):
			train(iterations=args.iterations)
			gc.collect()
			if not args.no_save: save(saveNet=True)

		print("\n" + str(len(M['trace'].baseConcepts)) + " concepts:", ", ".join(c.str(M['trace'], depth=1) for c in M['trace'].baseConcepts))
		addTask(M['state']['current_task'])
		checkForMistakes()	
		M['state']['current_task'] += 1
		gc.collect()
		if not args.no_save: save()
