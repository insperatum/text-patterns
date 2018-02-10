import os
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from robustfill import RobustFill, tensorToStrings, stringsToTensor
import random
import argparse
import regex
from datetime import datetime
vocab_size=128

# CUDA?
if torch.cuda.is_available(): torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Argument
default_name = "regex_dream"
if 'SLURM_JOB_NAME' in os.environ: default_name += "_" + os.environ['SLURM_JOB_NAME']
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default=default_name)
parser.add_argument('--cell_type', type=str, default="LSTM")
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--max_length', type=int, default=10)
parser.add_argument('--min_examples', type=int, default=4)
parser.add_argument('--max_examples', type=int, default=4)
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--embedding_size', type=int, default=128)
parser.add_argument('--mode', type=str, default="synthesis") # "synthesis" or "induction"
args = parser.parse_args()

# Load/create model
M = {}
M['net'] = net = RobustFill(hidden_size=args.hidden_size, embedding_size=args.embedding_size, cell_type=args.cell_type)
M['args'] = args
M['state'] = state = {'iteration':0, 'score':0, 'training_losses':[]}
opt = torch.optim.Adam(net.parameters(), lr=0.001)

# ------------------------------------------------------------------------------------------------

def getInstance(n_examples=1):
	"""
	Returns a single problem instance, as input/target strings
	"""
	inputs = ["foo" for i in range(n_examples)]
	target = "bar"

	if len(target)>args.max_length or any(len(x)>args.max_length for x in inputs):
		return getInstance(n_examples)
	else:
		return {'inputs':inputs, 'target':target}

def getBatch(b):
	"""
	Create a batch of problem instances, as tensors
	"""
	n_examples = random.randint(args.min_examples, args.max_examples)
	instances = [getInstance(n_examples) for i in range(b)]
	inputs = [stringsToTensor(vocab_size, [x['inputs'][j] for x in instances]) for j in range(n_examples)]
	target = stringsToTensor(vocab_size, [x['target'] for x in instances])
	return inputs, target


# Training
print("\nTraining...")
for state['iteration'] in range(state['iteration']+1, 200001):
	i = state['iteration']

	opt.zero_grad()
	inputs, target = getBatch(args.batch_size)
	score = net.score(inputs, target).mean()
	(-score).backward()
	opt.step()

	state['training_losses'].append(-score.data[0])
	state['score'] = score.data[0]
	print("Iteration %d Score %3.3f" % (i, state['score']))
	inputs = [stringsToTensor(net.v_input, ["foo"])] * 4
	outputs = tensorToStrings(net.v_output, net.sample(inputs))
	print(outputs)