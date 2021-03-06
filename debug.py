import loader
from propose import Proposal, evalProposal, getProposals, networkCache
import util

import torch

import os
import math
import argparse
import pregex as pre
from trace import RegexWrapper

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=max(('results/%s'%x for x in os.listdir('results') if x[-3:]==".pt"), key=os.path.getmtime)) #Most recent model
args = parser.parse_args()

print("Loading", args.model)
M = loader.load(args.model)

if 'net' in M and M['net'] is not None:
	if torch.cuda.is_available(): M['net'].cuda()
	net = M['net']

data, group_idxs, test_data = loader.loadData(M['args'].data_file, M['args'].n_examples, M['args'].n_tasks, M['args'].max_length)

trace = M['trace']
model = trace.model
