import loader
from propose import Proposal, evalProposal, getProposals, networkCache, getNetworkRegexes
import util

import torch

import os
import math
import argparse
import pregex as pre
from trace import RegexWrapper

for model in ('results/%s'%x for x in os.listdir('results') if x[-3:]==".pt"):
	print("Loading", args.model)
	M = loader.load(args.model)
	loader.saveRender(M)
