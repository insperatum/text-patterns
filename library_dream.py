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
import pickle
import model

# CUDA?
if torch.cuda.is_available(): torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Argument
default_name = "library_dream"
if 'SLURM_JOB_NAME' in os.environ: default_name += "_" + os.environ['SLURM_JOB_NAME']
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default=default_name)
parser.add_argument('--cell_type', type=str, default="LSTM")
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--max_length', type=int, default=15)
parser.add_argument('--data_file', type=str, default="./data/csv.p")
parser.add_argument('--min_examples', type=int, default=4)
parser.add_argument('--max_examples', type=int, default=4)
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--embedding_size', type=int, default=128)
parser.add_argument('--mode', type=str, default="synthesis") # "synthesis" or "induction"
args = parser.parse_args()

# Files to save:
os.makedirs("models", exist_ok = True)
os.makedirs("results", exist_ok = True)
modelfile = 'models/' + args.name + '.pt'
plotfile = 'results/' + args.name + '.png'

# Load/create model
try:
	print("Loading model ", modelfile)
	M = model.load(modelfile)
	net = M['net']
	args = M['args']
	state = M['state']
	opt = torch.optim.Adam(net.parameters(), lr=0.001)
	opt.load_state_dict(M['optstate'])

	data = [x['data'] for x in pickle.load(open(M['data_file'], 'rb'))]
	v_input = 128
	v_output = 128 + len(data)
except FileNotFoundError:
	data = [x['data'] for x in pickle.load(open(args.data_file, 'rb'))]
	v_input = 128
	v_output = 128 + len(data)

	M = {}
	M['net'] = net = RobustFill(v_input=v_input, v_output=v_output, hidden_size=args.hidden_size, embedding_size=args.embedding_size, cell_type=args.cell_type)
	M['args'] = args
	M['state'] = state = {'iteration':0, 'score':0, 'training_losses':[]}
	M['data_file'] = args.data_file
	M['model_file'] = modelfile
	opt = torch.optim.Adam(net.parameters(), lr=0.001)

# Data

lib_chars = ''.join([chr(128+i) for i in range(len(data))])
def lib_sample(char):
	idx = ord(char)-128
	return random.choice(data[idx])


# ------------------------------------------------------------------------------------------------

def getInstance(n_examples=1):
	"""
	Returns a single problem instance, as input/target strings
	"""
	r = regex.new(lib_chars)

	inputs = [regex.sample(r, lib_sample) for i in range(n_examples)]
	if args.mode == "synthesis":
		target = r
	elif args.mode == "induction":
		target = regex.sample(r, lib_sample)

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
	inputs = [stringsToTensor(v_input, [x['inputs'][j] for x in instances]) for j in range(n_examples)]
	target = stringsToTensor(v_output, [x['target'] for x in instances])
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
	if i>0 and i%200==0:
		plt.clf()
		plt.plot(range(1, i+1), state['training_losses'])
		plt.xlim(xmin=0, xmax=i)
		plt.ylim(ymin=0, ymax=20)
		plt.xlabel('iteration')
		plt.ylabel('NLL')
		plt.savefig(plotfile)
		M['optstate'] = opt.state_dict()
		print("Saving...")
		model.save(M)