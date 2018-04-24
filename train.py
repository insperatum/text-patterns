import os
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random
import argparse
import gc

from datetime import datetime
import pickle
import copy
import math
import numpy as np
from scipy import stats
import time
import string
from collections import Counter, namedtuple
import torch.multiprocessing as mp
import queue

from model import RegexModel
import pregex as pre
from trace import Trace, RegexWrapper
from pinn import RobustFill
import loader
from propose import Proposal, evalProposal, getProposals, networkCache


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--fork', type=str, default=None)
parser.add_argument('--data_file', type=str, default="./data/csv.p")
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--min_examples', type=int, default=2)
parser.add_argument('--max_examples', type=int, default=4)
parser.add_argument('--max_length', type=int, default=15) #maximum length of inputs or targets
parser.add_argument('--min_iterations', type=int, default=50) #minimum number of training iterations before next concept

parser.add_argument('--cell_type', type=str, default="LSTM")
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--embedding_size', type=int, default=128)

parser.add_argument('--n_tasks', type=int, default=40) #Per max_length
parser.add_argument('--skip_tasks', type=int, default=0)
parser.add_argument('--n_examples', type=int, default=500)

model_default_params = {'alpha':0.01, 'geom_p':0.01, 'pyconcept_alpha':1, 'pyconcept_d':0.5}
parser.add_argument('--alpha', type=float, default=None) #p(reference concept) proportional to #references, or to alpha if no references
parser.add_argument('--geom_p', type=float, default=None) #probability of adding another concept (geometric)
parser.add_argument('--pyconcept_alpha', type=float, default=None)
parser.add_argument('--pyconcept_d', type=float, default=None)

parser.add_argument('--helmholtz_dist', type=str, default="uniform") #During sleep, sample concepts from true weighted dist(default) or uniform
parser.add_argument('--regex-primitives', dest='regex_primitives', action='store_true')

parser.add_argument('--train_first', type=int, default=0)
parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--no-network', dest='no_network', action='store_true')
parser.add_argument('--no-cuda', dest='no_cuda', action='store_true')

parser.set_defaults(debug=False, no_cuda=False, regex_primitives=False, no_network=False)
args = parser.parse_args()
if args.fork is None:
	for k,v in model_default_params.items():
		if getattr(args,k) is None: setattr(args, k, v)

character_classes=[pre.dot, pre.d, pre.s, pre.w, pre.l, pre.u] if args.regex_primitives else [pre.dot]

# ----------- Network training ------------------
# Sample
def getInstance(n_examples):
	"""
	Returns a single problem instance, as input/target strings
	"""
	while True:
		r = M['trace'].model.sampleregex(M['trace'], conceptDist = args.helmholtz_dist)
		target = r.flatten()
		inputs = ([r.sample(M['trace']) for i in range(n_examples)],)
		if len(target)<args.max_length and all(len(x)<args.max_length for x in inputs): break
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
	if M['state']['iteration']%10==0: print("Iteration %d" % M['state']['iteration'], "| Network loss: %2.2f" % M['state']['network_losses'][-1])
	networkCache.clear()
	return network_score

def train(toConvergence=False, iterations=None, saveEvery=2000):
	from_iteration = M['state']['task_iterations'][-1] if M['state']['task_iterations'] else 0
	while True:
		if toConvergence:
			window_size = args.min_iterations
			min_grad=2e-3
			if len(M['state']['network_losses']) <= from_iteration + window_size:
				networkStep()
			else:
				window = M['state']['network_losses'][-window_size:]
				regress = stats.linregress(range(window_size), window)
				regress_slope = stats.linregress(range(window_size), [window[i] + min_grad*i for i in range(len(window))])
				p_ratio = regress.pvalue / regress_slope.pvalue
				if p_ratio < 2 and not args.debug: 
					networkStep()
				else:
					break #Break when converged
		else:
			if len(M['state']['network_losses']) <= from_iteration + iterations:
				networkStep()
			else:
				break

		if len(M['state']['network_losses']) % saveEvery == 0:
			loader.save(M)


# ----------- Solve a task ------------------
def onCounterexamples(queueProposal, proposal, counterexamples, p_valid, kinkscore=None):
	if p_valid>0.5 and proposal.depth==0 and (kinkscore is None or kinkscore < 0.6):
		counterexamples_unique = list(set(counterexamples))

		#Retry by including counterexamples in support set
		sampled_counterexamples = np.random.choice(counterexamples_unique, size=min(len(counterexamples_unique), 5), replace=False)
		counterexample_proposals = getProposals(M['net'] if not args.no_network else None, proposal.trace, proposal.examples + tuple(sampled_counterexamples), depth=proposal.depth+1)
		for counterexample_proposal in counterexample_proposals[:4]:
			queueProposal(counterexample_proposal)
		
		#Deal with counter examples separately (with Alt)
		sampled_counterexamples = np.random.choice(counterexamples, size=min(len(counterexamples), 5), replace=False)
		counterexample_proposals = getProposals(M['net'] if not args.no_network else None, proposal.trace, sampled_counterexamples, depth=proposal.depth+1)
		for counterexample_proposal in counterexample_proposals[:4]: 
			trace, concept = counterexample_proposal.trace.addregex(pre.Alt(
				[RegexWrapper(proposal.concept), RegexWrapper(counterexample_proposal.concept)], 
				ps = [p_valid, 1-p_valid]))
			new_proposal = Proposal(proposal.depth+1, proposal.examples + tuple(sampled_counterexamples), trace, concept, None, None, None)
			print("Adding proposal", new_proposal.concept.str(new_proposal.trace), "for counterexamples:", sampled_counterexamples, "kink =", kinkscore, flush=True)
			queueProposal(new_proposal)
		


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
		solution = evalProposal(proposal, task, onCounterexamples=lambda *args: q_counterexamples.put(args), doPrint=False, task_idx=task_idx)
		nEvaluated += 1
		if solution.valid:
			solutions.append(solution)
			print("(Worker %d)"%worker_idx, "Score: %3.3f"%(solution.final_trace.score - init_trace.score), "(prior %3.3f + likelihood %3.3f):"%(solution.trace.score - init_trace.score, solution.final_trace.score - solution.trace.score), proposal.concept.str(proposal.trace), flush=True)
		else:
			print("(Worker %d)"%worker_idx, "Failed:", proposal.concept.str(proposal.trace), flush=True)
		
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

	proposals = []
	proposal_strings_sofar = [] #TODO: this better. Want to avoid duplicate proposals. For now, just using string representation to check...

	def queueProposal(proposal): #add to evaluation queue
		proposal = proposal.strip() #Remove any evaluation data
		proposal_string = proposal.concept.str(proposal.trace, depth=-1) 
		if proposal_string not in proposal_strings_sofar:
			q_proposals.put(proposal)
			proposal_strings_sofar.append(proposal_string)

	for i in range(10 if not args.debug else 3):
		num_examples = random.randint(args.min_examples, args.max_examples)
		examples = list(np.random.choice(
			list(example_counter.keys()),
			size=min(num_examples, len(example_counter)),
			p=np.array(list(example_counter.values()))/sum(example_counter.values()),
			replace=True))
		pre_trace = M['trace']
		new_proposals = getProposals(M['net'] if not args.no_network else None, pre_trace, examples)
		for proposal in new_proposals:
			queueProposal(proposal)

	n_workers = max(1, cpus-1)

	q_counterexamples = manager.Queue()
	q_solutions = manager.Queue()
	l_active = manager.list([True for _ in range(n_workers)])
	for p in proposals:
		q_proposals.put(p)
	for worker_idx in range(n_workers):
		mp.Process(target=cpu_worker, args=(worker_idx, M['trace'], q_proposals, q_counterexamples, q_solutions, l_active, task_idx, data[task_idx])).start()

	while any(l_active) or not q_counterexamples.empty():
		try:
			counterexample_args = q_counterexamples.get(timeout=0.1)
			onCounterexamples(queueProposal, *counterexample_args)
		except queue.Empty:
			if not args.no_network: networkStep()

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
	M['trace'] = accepted.final_trace
	M['task_observations'][task_idx] = accepted.observations
	refreshVocabulary()
	M['state']['task_iterations'].append(M['state']['iteration'])
	print("Accepted proposal: " + accepted.concept.str(accepted.trace) + "\nScore:" + str(accepted.final_trace.score) + "\n")



# -----------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
	mp.set_start_method('spawn')

	# Compute
	if "SLURM_CPUS_PER_TASK" in os.environ: cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
	else: cpus = 1
	print("Running on %d CPUs" % cpus)


	default_vocabulary = list(string.printable) + \
		[pre.OPEN, pre.CLOSE, pre.String, pre.Concat, pre.Alt, pre.KleeneStar, pre.Plus, pre.Maybe] + \
		character_classes

	# Files to save:
	save_to = "results/"
	if not os.path.exists(save_to): os.makedirs(save_to)
	modelfile = save_to + "model.pt"
	use_cuda = torch.cuda.is_available() and not args.no_cuda

	# ------------- Load Model & Data --------------
	# Data
	data = loader.loadData(args.data_file, args.n_examples, args.n_tasks, args.max_length)

	# Model
	try:
		M = loader.load(modelfile)
		print("Loaded model ", modelfile)
		M['args'] = args

	except FileNotFoundError:
		if args.fork is not None:
			M = loader.load(args.fork, use_cuda)
			M['args'] = args
			print("Forked model", args.fork)
			for param in model_default_params:
				val = getattr(args, param)
				if val is not None:
					setattr(M['trace'].model, param, val)
					print("set model." + str(param) + "=" + str(val))
		else:
			M = {}
			M['state'] = {'iteration':0, 'current_task':0, 'network_losses':[], 'task_iterations':[]}
			M['net'] = net = RobustFill(input_vocabularies=[string.printable], target_vocabulary=default_vocabulary,
									 hidden_size=args.hidden_size, embedding_size=args.embedding_size, cell_type=args.cell_type)
			M['args'] = args
			M['task_observations'] = [[] for d in range(len(data))]
			M['trace'] = Trace(model=RegexModel(
				character_classes=character_classes,
				alpha=args.alpha,
				geom_p=args.geom_p,
				pyconcept_alpha=args.pyconcept_alpha,
				pyconcept_d=args.pyconcept_d))
			M['trace'], init_concept = M['trace'].addregex(pre.dot)
			print("Created new model")
		M['data_file'] = args.data_file
		M['save_to'] = save_to

	if use_cuda: M['net'].cuda()

	print("\nTraining...")
	refreshVocabulary()
	if use_cuda:  M['net'].cuda()

	def save():
		print("Saving...")
		if M['state']['current_task']%1==0: loader.saveCheckpoint(M)
		loader.saveRender(M)
		loader.save(M)
		print("Saved.")

	if args.train_first > 0: train(iterations=args.train_first)

	for i in range(M['state']['current_task'], len(data)):
		if not args.no_network: train(toConvergence=True)
		gc.collect()
		save()

		print("\n" + str(len(M['trace'].baseConcepts)) + " concepts:", ", ".join(c.str(M['trace'], depth=1) for c in M['trace'].baseConcepts))
		addTask(M['state']['current_task'])
		M['state']['current_task'] += 1
		gc.collect()
		save()
