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
parser.add_argument('--model', type=str, default=max(('results/%s'%x for x in os.listdir('results') if x[-3:]==".pt"), key=os.path.getmtime)) #Most recent model
args = parser.parse_args()

print("Loading", args.model)
M = loader.load(args.model)
if torch.cuda.is_available(): M['net'].cuda()
print("Loading data")
data, group_idxs, test_data = loader.loadData(M['args'].data_file, M['args'].n_examples, M['args'].n_tasks, M['args'].max_length)

def conditionalProbability(examples_support, examples_test): #p(examples_test | examples_support)
	proposals = getProposals(M['net'], M['trace'], examples_support, modes=("regex",))

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


hits=0
misses=0
for i in range(99999):
	print("-"*20, "\n")

	classes = [random.choice(test_data) for _ in range(5)]
	exampless = [list(np.random.choice(X, size=3)) for X in classes]

	print("Support Sets:")
	for examples in exampless: print(examples)
	
	examples_test = [random.choice(classes[0])]
	print("Test:")
	print(examples_test[0])
	print()

	scores = [conditionalProbability(examples, examples_test) for examples in exampless]
	print("Scores:", scores)

	confidence = math.exp(max(scores) - util.logsumexp(scores))
	if max(scores) == scores[0]:
		print("HIT (confidence %2.2f%%)" % (confidence*100))
		hits += 1
	else:
		print("MISS (confidence %2.2f%%)" % (confidence*100))
		misses += 1

	print("Accuracy: %2.2f%%" % (hits/(hits+misses) * 100))
