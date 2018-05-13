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

		totalJoint = util.logsumexp([x.final_trace.score for x in proposals])
		probs = [math.exp(x.final_trace.score - totalJoint) for x in proposals]
		posteriorConcepts = np.random.choice(range(len(proposals)), size=3, p=probs)
		samples = []
		for i in posteriorConcepts:
			for j in range(1000):
				s = proposals[i].concept.sample(proposals[i].trace)
				if s not in examples and s not in samples:
					samples.append(s)
					break
		print(examples, "; ".join(samples))
