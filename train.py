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

import model
import pregex as pre
from trace import Trace, RegexWrapper
from pinn import RobustFill
import loader


# Arguments
default_name = os.path.basename(__file__)
if 'SLURM_JOB_NAME' in os.environ: default_name += "_" + os.environ['SLURM_JOB_NAME']
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default=default_name)
parser.add_argument('--fork', type=str, default=None)
parser.add_argument('--data_file', type=str, default="./data/csv.p")
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--min_examples', type=int, default=1)
parser.add_argument('--max_examples', type=int, default=5)

parser.add_argument('--cell_type', type=str, default="LSTM")
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--embedding_size', type=int, default=128)
# parser.add_argument('--retrain_iterations', type=int, default=250)

parser.add_argument('--n_tasks', type=int, default=40) #Per max_length
parser.add_argument('--skip_tasks', type=int, default=0)
parser.add_argument('--n_examples', type=int, default=500)
# parser.add_argument('--pretrain_iterations', type=int, default=5000)

parser.add_argument('--demo', dest='demo', action='store_true')
parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--no-train', dest='no_train', action='store_true')
parser.add_argument('--no-cuda', dest='no-cuda', action='store_true')
parser.set_defaults(demo=False, debug=False, no_train=False, no_cuda=False)
args = parser.parse_args()


# ----------- Network training ------------------
# Sample
def getInstance(n_examples, pConcept):
	"""
	Returns a single problem instance, as input/target strings
	"""
	while True:
		r = model.sampleregex(M['trace'], model.pConcept)
		target = r.flatten()
		inputs = ([r.sample(M['trace']) for i in range(n_examples)],)
		if len(target)<25 and all(len(x)<25 for x in inputs): break
	return {'inputs':inputs, 'target':target}

def getBatch(batch_size):
	"""
	Create a batch of problem instances, as tensors
	"""
	n_examples = random.randint(args.min_examples, args.max_examples)
	instances = [getInstance(n_examples, model.pConcept) for i in range(batch_size)]
	inputs = [x['inputs'] for x in instances]
	target = [x['target'] for x in instances]
	return inputs, target

# SGD
def refreshRobustFill():
	#Make sure network/optimiser are updated to current vocabulary.
	M['net'] = M['net'].with_target_vocabulary(default_vocabulary + M['trace'].baseConcepts)

networkCache = {}

def networkStep():
	inputs, target = getBatch(args.batch_size)
	network_score = M['net'].optimiser_step(inputs, target)

	M['state']['network_losses'].append(-network_score)
	M['state']['iteration'] += 1
	if M['state']['iteration']%10==0: print("Iteration %d" % M['state']['iteration'], "| Network loss: %2.2f" % M['state']['network_losses'][-1])

	networkCache.clear()
	return network_score

def trainToConvergence():
	scores = []
	while True:
		scores.append(networkStep())

		if len(scores)>2:
			window = scores[-50:]
			regress = stats.linregress(range(len(window)), window)
			regress_slope = stats.linregress(range(len(window)), [window[i] - 0.005*i for i in range(len(window))])
			p_ratio = regress.pvalue / regress_slope.pvalue
			# print("p ratio ", p_ratio)
			if p_ratio > 1.2: 
				break #Break when converged


# ----------- Proposals ------------------

Proposal = namedtuple("Proposal", ["depth", "examples", "trace", "concept"]) #start with depth=0, increase depth when triggering a new proposal
def evalProposal(proposal, examples, onCounterexamples=None, doPrint=False):
	if proposal.trace.score == float("-inf"): #Zero probability under prior
		return None

	trace, observations, counterexamples, p_valid = proposal.trace.observe_all(proposal.concept, examples)

	if trace is None:
		if onCounterexamples is not None:
			if doPrint: print(proposal.concept.str(proposal.trace), "failed on", counterexamples)
			onCounterexamples(proposal, counterexamples, p_valid)
		return None
	else:
		if onCounterexamples is not None:
			scores = []
			for example in examples:
				single_example_trace, observation = proposal.trace.observe(proposal.concept, example)
				scores.append(single_example_trace.score - proposal.trace.score)

			if min(scores) != max(scores):
				values_zscores = list(zip(examples, (np.array(scores)-np.mean(scores))/np.std(scores)))
				min_value, min_zscore = min(values_zscores, key=lambda x: x[1])
				
				zscore_threshold=-2
				outliers = [value for value, zscore in values_zscores if zscore<zscore_threshold]
				if outliers:
					p_valid = 1-len(outliers)/len(examples)
					# print(proposal.concept.str(proposal.trace) + " got outliers", outliers, "p_valid=", p_valid)
					onCounterexamples(proposal, list(set(outliers)), p_valid)

		# onCounterexamples(proposal, )
		if doPrint: print(proposal.concept.str(proposal.trace), "got score", trace.score-proposal.trace.score)
		return {"trace":trace, "observations":observations, "concept":proposal.concept}

def getProposals(current_trace, examples, depth=0): #Includes proposals from network, and proposals on existing concepts
	examples = tuple(sorted(examples)[:10]) #Hashable for cache. Up to 10 input examples
	isCached = examples in networkCache
	# if not isCached: print("getProposals(", ", ".join(examples), ")")
	lookup = {concept: RegexWrapper(concept) for concept in current_trace.baseConcepts}
	if examples in networkCache:
		regex_count = networkCache[examples]
	else:
		regex_count = Counter()
		for i in range(10):
			inputs = [examples] * 500
			outputs = M['net'].sample(inputs)
			for o in outputs:
				try:
					r = pre.create(o, lookup=lookup)
					regex_count[r] += 1
				except pre.ParseException:
					pass
		networkCache[examples] = regex_count

	network_regexes = sorted(regex_count, key=regex_count.get, reverse=True)
	# if not isCached: print("  RobustFill:  ", ", ".join(str(r) for r in network_regexes[:10]))
	proposals = [Proposal(depth, examples, *current_trace.addregex(r)) for r in network_regexes] + \
		[Proposal(depth, examples, current_trace, c) for c in current_trace.baseConcepts] + \
		[Proposal(depth, examples, *current_trace.addregex(
			pre.String(examples[0]) if len(examples)==1 else pre.Alt([pre.String(x) for x in examples])))] #Exactly the examples

	evals = [evalProposal(proposal, examples) for proposal in proposals]
	scores = {proposals[i]:evals[i]['trace'].score for i in range(len(proposals)) if evals[i] is not None}
	proposals = sorted(scores.keys(), key=lambda proposal:-scores[proposal])
	proposals = proposals[:10]

	crp_proposals = []
	for proposal in proposals:
		new_trace, new_concept = proposal.trace.addCRP(proposal.concept)
		crp_proposals.append(Proposal(depth, examples, new_trace, new_concept))

	# if not isCached: print("  Proposals:", ", ".join(proposal.concept.str(proposal.trace) for proposal in proposals))
	return proposals + crp_proposals, scores

def onCounterexamples(q_proposals, proposal, counterexamples, p_valid):
	if p_valid>0.5 and proposal.depth==0:
		# print("Got counterexamples:", counterexamples)
		#Deal with counter examples separately (with Alt)
		counterexample_proposals, scores = getProposals(proposal.trace, counterexamples, depth=proposal.depth+1)
		for counterexample_proposal in counterexample_proposals: 
			trace, concept = counterexample_proposal.trace.addregex(pre.Alt(
				[RegexWrapper(proposal.concept), RegexWrapper(counterexample_proposal.concept)], 
				ps = [p_valid, 1-p_valid]))
			q_proposals.put(Proposal(proposal.depth+1, proposal.examples + tuple(counterexamples), trace, concept))
		
		#Retry by including counterexamples in support set
		counterexample_proposals, scores = getProposals(proposal.trace, proposal.examples + tuple(counterexamples), depth=proposal.depth+1)
		for counterexample_proposal in counterexample_proposals:
			q_proposals.put(counterexample_proposal)

def cpu_worker(worker_idx, q_proposals, q_counterexamples, q_solutions, task):
	solutions = []
	while(True):
		try:
			proposal = q_proposals.get(timeout=1)
			print("Worker", worker_idx, "evaluating:", proposal.concept.str(proposal.trace))
			solution = evalProposal(proposal, task, onCounterexamples=lambda *args: q_counterexamples.put(args), doPrint=False)
			if solution is not None: q_solutions.put(solution)
		except queue.Empty: break
	q_counterexamples.put(None)

def addTask(task_idx):
	print("\n" + "*"*40 + "\nAdding task %d (n=%d)" % (task_idx, len(data[task_idx])))
	print("Task: " + ", ".join(list(set(data[task_idx]))))

	example_counter = Counter(data[task_idx])
	# q_proposals = mp.Queue()
	
	proposals = []
	proposal_strings = [] #TODO: this better. Want to avoid duplicate proposals. For now, just using string representation to check
	for i in range(10):
		num_examples = random.randint(1, min(len(example_counter), args.max_examples))
		examples = list(np.random.choice(
			list(example_counter.keys()),
			size=min(num_examples, len(example_counter)),
			p=np.array(list(example_counter.values()))/sum(example_counter.values()),
			replace=False))
		pre_trace = M['trace']
		new_proposals, scores = getProposals(pre_trace, examples)
		for proposal in new_proposals:
			proposal_string = proposal.concept.str(proposal.trace) 
			if proposal_string not in proposal_strings:
				proposals.append(proposal)
				proposal_strings.append(proposal_string)

	q_proposals = mp.Manager().Queue()
	q_counterexamples = mp.Manager().Queue()
	q_solutions = mp.Manager().Queue()
	for p in proposals: q_proposals.put(p)
	for worker_idx in range(cpus-1):
		mp.Process(target=cpu_worker, args=(worker_idx, q_proposals, q_counterexamples, q_solutions, data[task_idx])).start()

	nCompleted = 0
	while(nCompleted < cpus-1):
		try:
			counterexample_args = q_counterexamples.get(timeout=1)
			if counterexample_args is None: nCompleted += 1
			else: onCounterexamples(q_proposals, *counterexample_args)
		except queue.Empty:
			networkStep()

	q_solutions.put(None)
	solutions = []
	while True:
		x = q_solutions.get()
		if x is None: break
		else: solutions.append(x)
	# pool = mp.Pool(cpus-1)
	# result = pool.map(worker, ((worker_idx, q_proposals, q_counterexamples, data[task_idx], worker_net) for worker_idx in range(cpus)))

	# nEvaluated = sum(nEvaluated for nEvaluated, solutions in result)
	# solutions = [solution for nEvaluated, solutions in result for solution in solutions]
	# print("Evaluated", nEvaluated, "proposals", "(%d solutions)"%len(solutions))
	print(len(solutions), "solutions")
	accepted = max(solutions, key=lambda p:p['trace'].score)
	M['trace'] = accepted['trace']
	M['task_observations'][task_idx] = accepted['observations']
	refreshRobustFill()
	print("Accepted proposal: " + accepted['concept'].str(accepted['trace']) + "\n")




# -----------------------------------------------------------------------------------------------------------------------------------------------



if __name__ == "__main__":
	mp.set_start_method('spawn')

	# Compute
	if "SLURM_CPUS_PER_TASK" in os.environ: cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
	else: cpus = 1
	print("Running on %d CPUs" % cpus)


	default_vocabulary = list(string.printable) + \
		[pre.OPEN, pre.CLOSE, pre.String, pre.Concat, pre.Alt, pre.KleeneStar, pre.Plus, pre.Maybe] + \
		model.character_classes

	# Files to save:
	os.makedirs("models", exist_ok = True)
	os.makedirs("results", exist_ok = True)
	modelfile = 'models/' + args.name + '.pt'
	plotfile = 'results/' + args.name + '.png'
	use_cuda = torch.cuda.is_available() and not args.no_cuda

	# ------------- Load Model & Data --------------
	# Data
	data = loader.loadData(args.data_file, args.n_examples, args.n_tasks)

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
		else:
			M = {}
			M['state'] = {'iteration':0, 'current_task':-1, 'network_losses':[]}
			M['net'] = net = RobustFill(input_vocabularies=[string.printable], target_vocabulary=default_vocabulary,
									 hidden_size=args.hidden_size, embedding_size=args.embedding_size, cell_type=args.cell_type)
			M['args'] = args
			M['task_observations'] = [[] for d in range(len(data))]
			M['trace'] = Trace()
			print("Created new model")
		M['data_file'] = args.data_file
		M['model_file'] = modelfile
		M['results_file'] = 'results/' + args.name + '.txt'

	if use_cuda: M['net'].cuda()



	if args.demo: # ------------ Demo -------------------
		i=0
		while True:
			print("-"*20, "\n")
			i += 1
			if i==1:
				examples = ["bar", "car", "dar"]
				print("Using examples:")
				for e in examples: print(e)
				print()
			else:
				print("Please enter examples:")
				examples = []
				nextInput = True
				while nextInput:
					s = input()
					if s=="":
						nextInput=False
					else:
						examples.append(s)

			proposals, scores = getProposals(M['trace'], examples)
			for proposal in sorted(proposals, key=lambda proposal:scores[proposal], reverse=True):
				print("\n%5.2f: %s" % (scores[proposal], proposal.concept.str(proposal.trace)))
				for i in range(3): print("  " + proposal.concept.sample(proposal.trace))
				
	else: # ------------------ Training -------------------
		print("\nTraining...")
		# saveEvery = 60 #seconds
		# nextSave = time.time() + saveEvery

		def save():
			# global nextSave
			plt.clf()
			plt.plot(range(1, M['state']['iteration']+1), M['state']['network_losses'])
			plt.xlim(xmin=0, xmax=M['state']['iteration']+1)
			# plt.ylim(ymin=0, ymax=25)
			plt.xlabel('iteration')
			plt.ylabel('NLL')
			plt.savefig(plotfile)
			print("Saving...")
			loader.save(M)
			# nextSave=time.time()+saveEvery

		refreshRobustFill()

		if M['state']['current_task'] == -1:
			if not args.no_train: trainToConvergence()
			M['state']['current_task'] += 1
			loader.saveCheckpoint(M)
			save()

		for i in range(M['state']['current_task'], len(data)+1):
			print("\n" + str(len(M['trace'].baseConcepts)) + " concepts:", ", ".join(c.str(M['trace']) for c in M['trace'].baseConcepts))
			addTask(M['state']['current_task'])
			gc.collect()

			if not args.no_train: trainToConvergence()
			M['state']['current_task'] += 1
			loader.saveCheckpoint(M)
			save()