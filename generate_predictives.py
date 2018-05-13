import loader
from propose import getProposals
import util

import numpy as np
import torch

import os
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=max(('results/%s'%x for x in os.listdir('results') if x[-3:]==".pt"), key=os.path.getmtime)) #Most recent model
parser.add_argument('--slow', dest='slow', action='store_const', const=True)
parser.set_defaults(slow=False)
args = parser.parse_args()

for model in ('results/%s'%x for x in os.listdir('results') if x[-3:]==".pt" and 'no_net' not in x):
	print("\nModel:", model)
	M = loader.load(args.model)
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
		if args.slow:
			proposals = list(getProposals(M['net'], M['trace'], examples, nProposals=50, maxNetworkEvals=100, doPrint=False))
		else:
			proposals = list(getProposals(M['net'], M['trace'], examples, doPrint=False))
		j=0

		totalJoint = util.logsumexp([x.final_trace.score for x in proposals])
		probs = [math.exp(x.final_trace.score - totalJoint) for x in proposals]
		posteriorConceptSamples = np.random.choice(range(len(proposals)), size=3, p=probs)
