import loader
from propose import getProposals
import util
import random

import numpy as np
import torch

import os
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nProposals', type=int, default=10)
parser.add_argument('--model', type=str, default=max(('results/%s'%x for x in os.listdir('results') if x[-3:]==".pt"), key=os.path.getmtime)) #Most recent model
parser.add_argument('--shot', type=int, default=3)
parser.add_argument('--way', type=int, default=10)
args = parser.parse_args()

print("Loading", args.model)
M = loader.load(args.model)
if torch.cuda.is_available(): M['net'].cuda()
print("Loading data")
data, group_idxs, test_data = loader.loadData(M['args'].data_file, M['args'].n_examples, M['args'].n_tasks, M['args'].max_length)

def conditionalProbability(examples_support, examples_test): #p(examples_test | examples_support)
	proposals = getProposals(M['net'], M['trace'], examples_support, modes=("regex",), printTimes=False)

	#Normalise scores
	total_logprob = util.logsumexp([proposal.final_trace.score for proposal in proposals])

	for proposal in proposals:
		proposal.final_trace.score -= total_logprob
	
	#Observe new examples
	new_traces = []
	for proposal in proposals:
		trace, observations, counterexamples, p_valid = proposal.final_trace.observe_all(proposal.concept, examples_test)
		if trace is not None:
			new_traces.append(trace)

	#Conditional probability
	return util.logsumexp([trace.score for trace in new_traces])

prob_cache = {}
def p_ratio(examples_support, examples_test):
	examples_support = tuple(sorted(examples_support))
	examples_test = tuple(sorted(examples_test))
	examples_joint = tuple(sorted(examples_support + examples_test))

	if examples_support in prob_cache:
		logp_support = prob_cache[examples_support]
	else:
		proposals_support = getProposals(M['net'], M['trace'], examples_support, modes=("regex",), printTimes=False, nProposals=args.nProposals)
		logp_support = util.logsumexp([proposal.final_trace.score for proposal in proposals_support])
		prob_cache[examples_support] = logp_support
	
	if examples_test in prob_cache:
		logp_test = prob_cache[examples_test]
	else:
		proposals_test = getProposals(M['net'], M['trace'], examples_test, modes=("regex",), printTimes=False, nProposals=args.nProposals)
		logp_test = util.logsumexp([proposal.final_trace.score for proposal in proposals_test])
		prob_cache[examples_test] = logp_test

	if examples_joint in prob_cache:
		logp_joint = prob_cache[examples_joint]
	else:
		proposals_joint = getProposals(M['net'], M['trace'], examples_joint, modes=("regex",), printTimes=False, nProposals=args.nProposals)
		logp_joint = util.logsumexp([proposal.final_trace.score for proposal in proposals_joint])
		prob_cache[examples_joint] = logp_joint

	score = logp_joint - logp_test - logp_support
	return score

hits=0
misses=0
for i in range(99999):
	print("-"*20, "\n")
	
	classes = [random.choice(test_data) for _ in range(args.way)]
	exampless = [list(np.random.choice(X, size=args.shot)) for X in classes]

	print("Support Sets:")
	for examples in exampless: print(examples)
	
	examples_test = [random.choice(classes[0])]
	print("Test:")
	print(examples_test[0])
	print()

	#scores = [conditionalProbability(examples, examples_test) for examples in exampless]
	scores = [p_ratio(examples, examples_test) for examples in exampless]
	print("Scores:", scores)

	confidence = math.exp(max(scores) - util.logsumexp(scores))
	if max(scores) == scores[0]:
		print("HIT (confidence %2.2f%%)" % (confidence*100))
		hits += 1
	else:
		print("MISS (confidence %2.2f%%)" % (confidence*100))
		misses += 1

	print("Accuracy: %d out of %d (%2.2f%%)" % (hits, hits+misses, (hits/(hits+misses) * 100)))
