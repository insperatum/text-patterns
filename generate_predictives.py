import loader
from propose import getProposals, networkCache
import util

import numpy as np
import torch

import os
import math

defaultExamples = [
	("Y",),
	("-7 degrees",),
	("TX --> CA",),
	("iii: true",),
	("1.8E-12",),
	("$3.00/min",),
	("Thur at 14:00",)
]

models = list('results/%s'%x for x in os.listdir('results') if x[-3:]==".pt" and 'no_net' not in x and x != "model.pt")
models.sort(key=os.path.getmtime)
nSamples=3

results = {} #results[model][defaultExamples]
for model in models: 
	results[model]={}
#for model in ["results/model.pt"]:
	print("\nModel:", model)
	M = loader.load(model)
	if torch.cuda.is_available(): M['net'].cuda()
	networkCache.clear()

	for examples in defaultExamples:
		proposals = list(getProposals(M['net'], M['trace'], examples, nProposals=50, maxNetworkEvals=100, doPrint=False))

		proposals = [x for x in proposals if not(all(x.concept.sample(x.trace)==examples[0] for _ in range(100)))]
		totalJoint = util.logsumexp([x.final_trace.score for x in proposals])
		probs = [math.exp(x.final_trace.score - totalJoint) for x in proposals]
		samples = []
		if len(proposals)>0:
			#print(examples)
			#for p in sorted(proposals, key=lambda p: p.final_trace.score, reverse=True)[:5]:
			#	print(p.concept.str(p.trace), p.concept.sample(p.trace))
			k=0
			for _ in range(500):
				p = proposals[np.random.choice(range(len(proposals)), p=probs)]
				s = p.concept.sample(p.trace)
				if s not in examples and s not in samples:
					samples.append(s)
					k+=1
				if k==nSamples:
					break
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
		results[model][examples] = [x.replace("#", "\\#").replace("$", "\\$") for x in samples]

#print()
#for model in models:
#	print(results[model])
	
print("\n")
print("\\begin{tabular}{" + " ".join("l"*(len(models)+1)) + "}")
print("Input" + "".join(" & Stage " + str(i+1) for i in range(len(models))) + "\\\\ \hline")
for i in range(len(defaultExamples)):
	examples = defaultExamples[i]
	for j in range(nSamples):
		print(examples[j] if j<len(examples) else "", "".join(" & " + (results[model][examples][j] if j<len(results[model][examples]) else "") for model in models) + "\\\\")
	print("\hline")
print("\end{tabular}")
