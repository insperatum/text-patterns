import loader
from propose import getProposals
import util

import numpy as np
import torch

import os
import math

models = list('results/%s'%x for x in os.listdir('results') if x[-3:]==".pt" and 'no_net' not in x)
models.sort(key=os.path.getmtime)
for model in models: 
#for model in ["results/model.pt"]:
	print("\nModel:", model)
	M = loader.load(model)
	if torch.cuda.is_available(): M['net'].cuda()

	defaultExamples = [
		["F"],
		["-7 degrees"],
		["TX --> CA"],
		["iii: true"],
		["1.8E-12"],
		["$3.00/min"],
		["Thur at 14:00"]
	]
	for examples in defaultExamples:
		proposals = list(getProposals(M['net'], M['trace'], examples, nProposals=50, maxNetworkEvals=100, doPrint=False))

		proposals = [x for x in proposals if not(all(x.concept.sample(x.trace)==examples[0] for _ in range(100)))]
		totalJoint = util.logsumexp([x.final_trace.score for x in proposals])
		probs = [math.exp(x.final_trace.score - totalJoint) for x in proposals]
		samples = []
		k=0
		if len(proposals)>0:
			#print(examples)
			#for p in sorted(proposals, key=lambda p: p.final_trace.score, reverse=True)[:5]:
			#	print(p.concept.str(p.trace), p.concept.sample(p.trace))
			for _ in range(3):
				p = proposals[np.random.choice(range(len(proposals)), p=probs)]
				samples.append(p.concept.sample(p.trace))
				#print(p.concept.str(p.trace), p.concept.sample(p.trace))
			#for _ in range(500):
			#	i = np.random.choice(range(len(proposals)), p=probs)
			#	#print(proposals[i].concept.str(proposals[i].trace))
			#	for j in range(1000):
			#		s = proposals[i].concept.sample(proposals[i].trace)
			#		#if s not in examples and s not in samples:
			#		samples.append(s)
			#		k+=1
			#		break
			#	if k==3: break

		print(examples, "; ".join(samples))
