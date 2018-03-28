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

from datetime import datetime
import pickle
import copy
import math
import numpy as np
import time
import string
from collections import Counter, namedtuple
from multiprocessing import Pool

import model
import regex
from trace import Trace, RegexWrapper
from network import Network
import loader

# Compute
if torch.cuda.is_available(): torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Arguments
default_name = os.path.basename(__file__)
if 'SLURM_JOB_NAME' in os.environ: default_name += "_" + os.environ['SLURM_JOB_NAME']
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default=default_name)
parser.add_argument('--fork', type=str, default=None)
parser.add_argument('--data_file', type=str, default="./data/csv.p")
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--min_examples', type=int, default=3)
parser.add_argument('--max_examples', type=int, default=3)

parser.add_argument('--cell_type', type=str, default="LSTM")
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--embedding_size', type=int, default=128)
parser.add_argument('--retrain_iterations', type=int, default=200)

parser.add_argument('--n_tasks', type=int, default=100)
parser.add_argument('--skip_tasks', type=int, default=0)
parser.add_argument('--n_examples', type=int, default=100)
parser.add_argument('--pretrain_iterations', type=int, default=5000)

parser.add_argument('--demo', dest='demo', action='store_true')
parser.set_defaults(demo=False)
args = parser.parse_args()

# Files to save:
os.makedirs("models", exist_ok = True)
os.makedirs("results", exist_ok = True)
modelfile = 'models/' + args.name + '.pt'
plotfile = 'results/' + args.name + '.png'


# Network and Optimiser
default_vocabulary = list(string.printable) + \
	[regex.OPEN, regex.CLOSE, regex.String, regex.Concat, regex.Alt, regex.KleeneStar, regex.Plus, regex.Maybe] + \
	model.character_classes


# ------------- Load Model & Data --------------
# model
try:
	M = loader.load(modelfile)
	print("Loaded model ", modelfile)
	M['args'] = args

except FileNotFoundError:
	if args.fork is not None:
		M = loader.load(args.fork)
		M['args'] = args
	else:
		M = {}
		M['state'] = {'iteration':0, 'current_task':0, 'network_losses':[]}
		M['net'] = net = Network(input_vocabulary=string.printable, output_vocabulary=default_vocabulary,
								 hidden_size=args.hidden_size, embedding_size=args.embedding_size, cell_type=args.cell_type)
		M['args'] = args
	M['data_file'] = args.data_file
	M['model_file'] = modelfile
	M['results_file'] = 'results/' + args.name + '.txt'

#Data
def loadData(file):
	rand = np.random.RandomState()
	rand.seed(0)
	
	data = []
	tasks_unique = [] #unique elements for each task
	for x in pickle.load(open(file, 'rb')):
		elems_filtered = [elem for elem in x['data'] if len(elem)<25]
		if len(elems_filtered)==0: continue
		# rand.shuffle(elems_filtered)

		task = rand.choice(elems_filtered, size=min(len(elems_filtered), args.n_examples), replace=False).tolist()
		unique = sorted(list(set(task)))
		if unique not in tasks_unique: #No two tasks should have the same set of unique elements
			data.append(task)
			tasks_unique.append(unique)
	
	rand.shuffle(data)
	if args.n_tasks is not None:
		data = data[args.skip_tasks:args.n_tasks + args.skip_tasks]

	data = sorted(data, key=lambda examples: sum(len(x) for x in examples) / len(examples))
	return data

data = loadData(M['data_file'])
# print("Tasks:")
# for x in data[:100]:
# 	print(x[:10])


# ----------- Helmholtz ------------------

def getInstance(n_examples, pConcept):
	"""
	Returns a single problem instance, as input/target strings
	"""
	while True:
		r = model.sampleregex(M['trace'].baseConcepts, model.pConcept)
		target = r.flatten()
		inputs = [r.sample(M['trace']) for i in range(n_examples)]
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


# --------- Trace Initialisation & Proposals ---------------

def refreshNetwork():
	#Make sure network/optimiser are updated to current vocabulary.
	M['net'].set_output_vocabulary(default_vocabulary + M['trace'].baseConcepts)
	M['net'] = copy.deepcopy(M['net']) #Annoying to have to do this, but it's okay

def initialiseTrace():
	# print("Initialising trace...")
	M['task_observations'] = [[] for d in range(len(data))]
	M['trace'] = Trace()
	# for i in range(len(data)):
	# 	d = data[i]
	# 	t = M['task_observations'][i]
	# 	if i%20==0: print("Concept %d/%d (%d elements)" % (i, len(data), len(d)))
	# 	for x in d:
	# 		trace, observation = trace.observe(initialConcept, x)
	# 		t.append(observation)
	
	# print("Initialised trace with baseConcepts:", ", ".join(c.str(M['trace']) for c in M['trace'].baseConcepts))
	# refreshNetwork(M)

Proposal = namedtuple("Proposal", ["trace", "concept"])
def evalProposal(proposal, examples, onCounterexamples=None, doPrint=False):

	# if type(regex) is RegexWrapper:
	# 	if addCRP:
	# 		trace, concept = trace.addCRP(regex.concept)
	# 	else:
	# 		trace, concept = trace, regex.concept
	# else:
	# 	if addCRP:
	# 		trace, concept = trace.addCRPregex(regex)
	# 	else:
	# 		trace, concept = trace.addregex(regex)

	if proposal.trace.score == float("-inf"): #Zero probability under prior
		return None

	trace, observations, counterexamples, p_valid = proposal.trace.observe_all(proposal.concept, examples)

	if trace is None:
		if onCounterexamples is not None:
			if doPrint: print(proposal.concept.str(proposal.trace), "failed on", counterexamples)
			onCounterexamples(proposal, counterexamples, p_valid)
		return None
	else:
		if doPrint: print(proposal.concept.str(proposal.trace), "got score", trace.score-proposal.trace.score)
		return {"trace":trace, "observations":observations, "concept":proposal.concept}

networkCache = {}
def resetNetworkCache():
	networkCache.clear()

def networkStep():
	inputs, target = getBatch(args.batch_size)
	network_score = M['net'].optimiser_step(inputs, target)

	M['state']['network_losses'].append(-network_score)
	resetNetworkCache()

def getProposals(current_trace, examples): #Includes proposals from network, and proposals on existing concepts
	examples = tuple(sorted(examples)) #Hashable for cache
	isCached = examples in networkCache
	if not isCached: print("getProposals(", ", ".join(examples),")")
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
					r = regex.create(o, lookup=lookup)
					regex_count[r] += 1
				except regex.ParseException:
					pass
		networkCache[examples] = regex_count

	network_regexes = sorted(regex_count, key=regex_count.get, reverse=True)
	if not isCached: print("  Network:  ", ", ".join(str(r) for r in network_regexes[:10]))
	proposals = [Proposal(*current_trace.addregex(r)) for r in network_regexes] + [Proposal(current_trace, c) for c in current_trace.baseConcepts]

	evals = [evalProposal(proposal, examples) for proposal in proposals]
	scores = {proposals[i]:evals[i]['trace'].score for i in range(len(proposals)) if evals[i] is not None}
	proposals = sorted(scores.keys(), key=lambda proposal:-scores[proposal])
	proposals = proposals[:5]
	if not isCached: print("  Proposals:", ", ".join(proposal.concept.str(proposal.trace) for proposal in proposals))
	return proposals, scores

def addTask(task_idx):
	print("\nAdding task %d (n=%d)" % (task_idx, len(data[task_idx])))
	print("Examples: [" + ", ".join(list(set(data[task_idx]))[:5]) + (", ...]" if len(set(data[task_idx]))>5 else "]"))

	solutions = []
	nEvaluated = 0
	
	example_counter = Counter(data[task_idx])
	# most_common = sorted(example_counter, key=example_counter.get, reverse=True)
	for num_examples in range(1, 1+min(4, len(example_counter))): #use most common n examples in task
		# unique = list(set(data[task_idx]))
		# examples = list(np.random.choice(unique, size=min(num_examples, len(unique)), replace=False)) #example[proposal][i]
		# examples = most_common[:num_examples]
		examples = list(np.random.choice(
			list(example_counter.keys()),
			size=min(num_examples, len(example_counter)),
			p=np.array(list(example_counter.values()))/sum(example_counter.values()),
			replace=False))
		pre_trace = M['trace']#.unobserve_all(M['task_observations'][task_idx])

		proposals, scores = getProposals(pre_trace, examples)

		current = {"trace":M['trace'], "observations":M['task_observations'], "concept":None}
		
		
		counterproposals = []
		def gotCounterexamples(proposal, counterexamples, p_valid):
			print(proposal.concept.str(proposal.trace), "missed", ", ".join(counterexamples), "(p_valid=%.2f)"%p_valid)
			if p_valid>0.5:
				#Deal with counter examples separately (with Alt)
				counterexample_proposals, scores = getProposals(proposal.trace, counterexamples)
				for counterexample_proposal in counterexample_proposals: 
					trace, concept = counterexample_proposal.trace.addregex(regex.Alt(
						RegexWrapper(proposal.concept), 
						RegexWrapper(counterexample_proposal.concept)))
					counterproposals.insert(0, Proposal(trace, concept))
				
				#Retry by including counterexamples in support set
				counterexample_proposals, scores = getProposals(pre_trace, examples + counterexamples)
				for counterexample_proposal in counterexample_proposals:
					counterproposals.insert(0, counterexample_proposal)
		
		while len(proposals)>0 or len(counterproposals)>0:
			if len(proposals)>0:
				proposal = proposals.pop(0)
				onCounterexamples = gotCounterexamples
			else:
				proposal = counterproposals.pop(0)
				onCounterexamples = None
			p = evalProposal(proposal, data[task_idx], onCounterexamples=onCounterexamples, doPrint=False)
			nEvaluated += 1
			if p is not None:
				solutions.insert(0, p)
				trace, concept = proposal.trace.addCRP(proposal.concept)
				proposal = Proposal(trace, concept)
				p = evalProposal(proposal, data[task_idx], doPrint=False)
				nEvaluated += 1
				if p is not None:
					solutions.insert(0, p)

	print("Evaluated", nEvaluated, "proposals", "(%d solutions)"%len(solutions))

	accepted = max(solutions, key=lambda p:p['trace'].score)
	M['trace'] = accepted['trace']
	M['task_observations'][task_idx] = accepted['observations']
	refreshNetwork()
	print("Accepted proposal: " + accepted['concept'].str(accepted['trace']))








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
	saveEvery = 60 #seconds
	nextSave = time.time() + saveEvery

	def save():
		global nextSave
		plt.clf()
		plt.plot(range(1, M['state']['iteration']+1), M['state']['network_losses'])
		plt.xlim(xmin=0, xmax=M['state']['iteration']+1)
		# plt.ylim(ymin=0, ymax=25)
		plt.xlabel('iteration')
		plt.ylabel('NLL')
		plt.savefig(plotfile)
		print("Saving...")
		loader.save(M)
		nextSave=time.time()+saveEvery


	if 'trace' not in M: initialiseTrace()
	refreshNetwork()

	while M['state']['iteration'] < args.pretrain_iterations:
		networkStep()
		M['state']['iteration'] += 1
		print("Iteration", M['state']['iteration'], "Network loss:", M['state']['network_losses'][-1])
		if time.time() > nextSave: save()

	if M['state']['iteration'] == args.pretrain_iterations: loader.saveIteration(M)

	for M['state']['current_task'] in range(M['state']['current_task'], len(data)+1):
		addTask(M['state']['current_task'])
		print("\n" + str(len(M['trace'].baseConcepts)) + " concepts:", ", ".join(c.str(M['trace']) for c in M['trace'].baseConcepts))
		for i in range(args.retrain_iterations):
			networkStep()
			M['state']['iteration'] += 1
			print("Iteration", M['state']['iteration'], "Network loss:", M['state']['network_losses'][-1])
		if time.time() > nextSave: save()