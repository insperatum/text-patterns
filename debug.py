import loader
from propose import Proposal, evalProposal, getProposals, networkCache, getNetworkRegexes
import util

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

net = M['net']
trace = M['trace']
model = trace.model
