import loader
from propose import getProposals, getNetworkRegexes
import util

import torch

import os
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=max(('results/%s'%x for x in os.listdir('results') if x[-3:]==".pt"), key=os.path.getmtime)) #Most recent model
parser.add_argument('--slow', dest='slow', action='store_const', const=True)
parser.set_defaults(slow=False)
args = parser.parse_args()

print("Loading", args.model)
M = loader.load(args.model)
if torch.cuda.is_available(): M['net'].cuda()

print("Few shot (g)eneration, (c)lassification, or (n)etwork predictions?")
mode = input()


if mode.lower() in ["n", "network"]:
	for i in range(99999):
		print("-"*20, "\n")
		if i==0:
			examples = ["bar", "car", "dar"]
			print("Using examples:")
			for e in examples: print(e)
			print()
		else:
			print("Please enter examples (one per line):")
			examples = []
			nextInput = True
			while nextInput:
				s = input()
				if s=="":
					nextInput=False
				else:
					examples.append(s)

		j=0
		for r,count in getNetworkRegexes(M['net'], M['trace'], examples):
			s = r.str(lambda concept: concept.str(M['trace'], depth=-1))
			print("%3d: %s" %(count, s))
			j+=1
			if j==20: break	

if mode.lower() in ["g", "generation"]:
	defaultExamples = [
		["Thur at 14:00"],
		["MA -> TX"],
		["iv: No"],
		["1.8E-12"]
	]
	for i in range(99999):
		print("-"*20, "\n")
		if i<len(defaultExamples):
			examples = defaultExamples[i] 
			print("Using examples:")
			for e in examples: print(e)
			print()
		else:
			print("Please enter examples (one per line):")
			examples = []
			nextInput = True
			while nextInput:
				s = input()
				if s=="":
					nextInput=False
				else:
					examples.append(s)

		if args.slow:
			proposals = getProposals(M['net'], M['trace'], examples, nProposals=50, maxNetworkEvals=100)
		else:
			proposals = getProposals(M['net'], M['trace'], examples)
		j=0
		for proposal in proposals:
			print("\n%5.2f: %s" % (proposal.final_trace.score, proposal.concept.str(proposal.trace)))
			for _ in range(3): print("  " + proposal.concept.sample(proposal.trace))
			j+=1
			if j>5: break

elif mode.lower() in ["c", "classification"]:
	raise NotImplementedError() #which classification mode? Total correlation?
	def getExamples():
		examples = []
		nextInput = True
		while nextInput:
			s = input()
			if s=="":
				nextInput=False
			else:
				examples.append(s)
		return examples

	def conditionalProbability(examples_support, examples_test): #p(examples_test | examples_support)
		proposals = getProposals(M['net'], M['trace'], examples_support)

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


	for i in range(99999):
		print("-"*20, "\n")

		if i==0:
			examples1 = ["1", "2"]
			print("Examples of class 1:\n" + "\n".join(examples1) + "\n")
			examples2 = ["a", "b"]
			print("Examples of class 2:\n" + "\n".join(examples2) + "\n")
			examples_test = ["7"]
			print("Test Examples:\n" + "\n".join(examples_test) + "\n")
		else:
			print("Please enter examples of class 1 (one per line):")
			examples1 = getExamples()
			print("Please enter examples of class 2:")
			examples2 = getExamples()
			print("Please enter test examples:")
			examples_test = getExamples()

		score1 = conditionalProbability(examples1, examples_test)
		print("Class 1 posterior predictive:", score1)
		print()
		score2 = conditionalProbability(examples2, examples_test)
		print("Class 2 posterior predictive:", score2)

		print()
		prediction = "class 1" if score1>score2 else "class 2",
		confidence = math.exp(max(score1, score2) - util.logsumexp([score1, score2]))
		print("Prediction: %s"%prediction, "(confidence %2.2f%%)"%(confidence*100))
