import loader
from propose import Proposal, evalProposal, getProposals, networkCache

import torch

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--demo', dest='demo', action='store_true')
parser.set_defaults(demo=False)
args = parser.parse_args()

modelfile = max(('models/%s'%x for x in os.listdir('models')), key=os.path.getmtime) #Most recent model
M = loader.load(modelfile)
if torch.cuda.is_available(): M['net'].cuda()



if args.demo:
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

		proposals, scores = getProposals(M['net'], M['trace'], examples, include_crp=False)
		for proposal in proposals:
			print("\n%5.2f: %s" % (scores[proposal], proposal.concept.str(proposal.trace)))
			for _ in range(3): print("  " + proposal.concept.sample(proposal.trace))