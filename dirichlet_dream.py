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
import copy
import math
from collections import Counter
import numpy as np

data_maxsize=10000
expected_num_concepts = 100

# Todo
# Library:
#	When removing library item, delete other redundant items that are only referenced by it
#	Add dirichlet process
#	Allow proposals that reference items later in the library, but prevent cycles (importantly, prevent self-reference!)
# NN:
#	Work out real distribution, and train on it
#	Beam search
#	Train NN in parallel with proposals?
# Regex:
#	Rewrite properly:
#		First parse regex
#		Then maintain a queue of logprob + remainders + regex_continuation pairs? Something like that?
#		order of operations a|b+ == (a)|(b+)
#		then I can get rid of this arbitrary depth limit lib_depth


# Observation has {"task":_, "ancestors":_, "obs":_}
# library[concept_idx]["observations"] = [Observation1, Observation2, ...] such that Observationi["ancestors"]["-1"]==concept_idx
# Proposal has {"regex":_, "score_diff":_, "observations":_}

# CUDA?
if torch.cuda.is_available(): torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Argument
default_name = "dirichlet_dream"
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

# Load/create model+state
def createOptimiser(net):
	return torch.optim.Adam(net.parameters(), lr=0.001)
try:
	print("Loading model ", modelfile)
	M = model.load(modelfile)
	net = M['net']
	args = M['args']
	state = M['state']
	library = M['library']
	data_concepts = M['data_concepts']
	opt = createOptimiser(net)
	opt.load_state_dict(M['optstate'])

	rand = random.Random()
	rand.seed(0)
	data = [[rand.choice(x['data']) for i in range(data_maxsize)] for x in pickle.load(open(M['data_file'], 'rb'))]
	rand.shuffle(data)
	data_counts = [Counter(d) for d in data]
except FileNotFoundError:
	rand = random.Random()
	rand.seed(0)
	data = [[rand.choice(x['data']) for i in range(data_maxsize)] for x in pickle.load(open(args.data_file, 'rb'))]
	rand.shuffle(data)
	data_counts = [Counter(d) for d in data]

	v_input = 128
	v_output = 128 + 1 #Initialise library with size 1

	M = {}
	M['net'] = net = RobustFill(v_input=v_input, v_output=v_output, hidden_size=args.hidden_size, embedding_size=args.embedding_size, cell_type=args.cell_type)
	M['args'] = args
	M['state'] = state = {'iteration':0, 'score':0, 'training_losses':[]}
	M['data_file'] = args.data_file
	M['model_file'] = modelfile
	M['results_file'] = 'results/' + args.name + '.txt'
	M['library'] = library = [{'base':regex._["."]+regex._["*"], 'observations':[{"task":i, "ancestors":[0], "obs":obs} for i in range(len(data)) for obs in data[i]]}]
	M['data_concepts'] = data_concepts = [0 for i in range(len(data))] #For each data column, what concept does it belong to
	opt = createOptimiser(net)

# Data
lib_chars = ''.join([chr(128+i) for i in range(len(library))])
def lib_sample(char):
	idx = ord(char)-128
	return regex.sample(library[idx]['base'], lib_sample=lib_sample)
def lib_score(lib_char, s, lib_depth):
	idx = ord(lib_char)-128
	try:
		return regex.match(s, library[idx]['base'], lib_score=lib_score, mode="full", lib_depth=lib_depth)
	except regex.RegexException:
		return {"score":float("-inf"), "observations":None}
def char_map(char):
	idx=ord(char)-128
	return regex.humanreadable(library[idx]['base'], char_map)

# ------------------------------------------------------------------------------------------------

def getInstance(n_examples=1):
	"""
	Returns a single problem instance, as input/target strings
	"""
	
	# if random.random()<1/2:
	r = regex.new(lib_chars)
	# else:
	# 	lib_idx = random.randint(0, len(library)-1)
	# 	r = chr(lib_idx+128)
	inputs = [regex.sample(r, lib_sample=lib_sample) for i in range(n_examples)]

	if args.mode == "synthesis":
		target = r
	elif args.mode == "induction":
		target = regex.sample(r, lib_sample=lib_sample)

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
	inputs = [stringsToTensor(net.v_input, [x['inputs'][j] for x in instances]) for j in range(n_examples)]
	target = stringsToTensor(net.v_output, [x['target'] for x in instances])
	return inputs, target


def getPrior(r):
	return -math.log(40)*len(r)


def dirichlet_score(base, obs):
	alpha=1
	c = Counter(obs)

def getLikelihoodAndObservations(r, examples):
	"""
	Returns (score, observations, counterexample)
	"""
	cnt = Counter(examples)
	try:
		ll = 0
		observations = []
		for s in cnt:
			MAP = regex.match(s, r, lib_score=lib_score, mode="full")
			if MAP['score'] == float('-inf'):
				return float('-inf'), [], s
			ll += MAP['score'] * cnt[s]
			observations += [{"ancestors":[], "obs":s}] * cnt[s]
			observations += MAP['observations'] * cnt[s]
		return ll, observations, None
	except regex.RegexException:
		return float('-inf'), [], None

def refreshAdam(net):
	global opt
	new_opt = createOptimiser(net)
	for i in range(len(opt.param_groups)):
		opt.param_groups[i]['params'] = new_opt.param_groups[i]['params']

def makeProposalOnCol(task_idx, specificProposal=None):
	#Returns true if library has updated
	global lib_chars, net
	old_concept = data_concepts[task_idx]

	

	# Counterexample-driven:
	# examples = [list(np.random.choice(unique, size=4)) for _ in range(1000)] #example[proposal][i]
	# while True:
	# 	inputs = [stringsToTensor(net.v_input, [e[j] for e in examples]) for j in range(len(examples[0]))]
	# 	proposals = tensorToStrings(net.v_output, net.sample(inputs))
	# 	if len(examples[0])>=5: break
	# 	for j in range(len(proposals)):
	# 		_, _, counterexample = getLikelihoodAndObservations(proposals[j], data[task_idx])
	# 		examples[j].append( counterexample if counterexample is not None else random.choice(unique) )
	
	# Normal:
	if specificProposal is None:
		unique = list(set(data[task_idx]))
		examples = list(np.random.choice(unique, size=5)) #example[proposal][i]
		print("Examples:", examples)
		proposals = []
		for i in range(2):
			inputs = [stringsToTensor(net.v_input, [example]*500) for example in examples]
			proposals += tensorToStrings(net.v_output, net.sample(inputs))
		proposals += [chr(i+128) for i in range(len(library))]
		proposals = [r for r in proposals if r != chr(old_concept+128)] #Don't propose the same thing
		moreExamples = list(np.random.choice(unique, size=100))
		proposals = sorted(list(set(proposals)), key=lambda r: -(getPrior(r) + getLikelihoodAndObservations(r, moreExamples)[0]))[:1]
	else:
		proposals = [specificProposal]
		proposals = [r for r in proposals if r != chr(old_concept+128)] #Don't propose the same thing
	
	proposals = [{"regex":p} for p in proposals]
	old_ll, old_observations, _ = getLikelihoodAndObservations(library[old_concept]['base'], data[task_idx])
	for proposal in proposals:
		print("Proposal:", regex.humanreadable(proposal['regex'], char_map=char_map))
		if len(proposal['regex'])==1 and ord(proposal['regex'])>=128: #Existing concept
			new_concept = ord(proposal['regex'])-128
			new_ll, proposal['observations'], counterexample = getLikelihoodAndObservations(library[new_concept]['base'], data[task_idx])
			if counterexample is not None: print("Counterexample:", counterexample)
			for observation in proposal['observations']: observation['task']=task_idx
			score_diff = new_ll - old_ll
			if not any([observation['task'] != task_idx for observation in library[old_concept]['observations']]):
				score_diff -= getPrior(library[old_concept]['base'])
				score_diff += math.log(1-1/expected_num_concepts) #Geometric prior on length
			proposal['score_diff'] = score_diff

		else:
			new_ll, proposal['observations'], counterexample = getLikelihoodAndObservations(proposal['regex'], data[task_idx])
			if counterexample is not None: print("Counterexample:", counterexample)
			for observation in proposal['observations']: observation['task']=task_idx
			score_diff = new_ll - old_ll
			score_diff += getPrior(proposal['regex'])
			score_diff += math.log(1-1/expected_num_concepts) #Geometric prior on length
			if not any([observation['task'] != task_idx for observation in library[old_concept]['observations']]):
				score_diff -= getPrior(library[old_concept]['base'])
				score_diff -= math.log(1-1/expected_num_concepts) #Geometric prior on length
			proposal['score_diff'] = score_diff
	
	libraryUpdated=False
	if len(proposals)>0:
		best_proposal = max(proposals, key=lambda x:x['score_diff'])
		if best_proposal['score_diff']>0:
			if len(best_proposal['regex'])==1 and ord(best_proposal['regex'])>=128: #Existing concept
				print("Move to concept accepted:", regex.humanreadable(best_proposal['regex'], char_map=char_map))
				data_concepts[task_idx] = new_concept
				for observation in best_proposal['observations']: observation['ancestors'].insert(0, new_concept)
			else:
				print("New concept accepted", regex.humanreadable(best_proposal['regex'], char_map=char_map))
				library.append({'base':best_proposal['regex'], 'observations':[]})
				new_concept = len(library)-1
				data_concepts[task_idx] = new_concept
				for observation in best_proposal['observations']: observation['ancestors'].insert(0, new_concept)
				net.append_output()
				M['net'] = net = copy.deepcopy(net) #So weird to have to do this!
				refreshAdam(net)
				libraryUpdated=True


			for i in range(len(library)):
				library[i]['observations'] = [observation for observation in library[i]['observations'] if observation['task'] != task_idx]
				library[i]['observations'] += [observation for observation in best_proposal['observations'] if observation['task'] == task_idx and observation['ancestors'][-1]==i]

			if not any([observation['task'] != task_idx for observation in library[old_concept]['observations']]): #Old concept is unused
				print("Deleting concept", old_concept)
				library.pop(old_concept)
				for L in library:
					for i in range(128+old_concept, 128+len(library)):
						L['base'] = L['base'].replace(chr(i+1), chr(i))
					for observation in L['observations']:
						observation['ancestors'] = [x if x<old_concept else x-1 for x in observation['ancestors']]
				M['net'].remove_output(old_concept + 128)
				M['net'] = net = copy.deepcopy(net) #So weird to have to do this!
				refreshAdam(net)
			lib_chars = ''.join([chr(128+i) for i in range(len(library))])
	return libraryUpdated

# Training
print("\nTraining...")
pretrainUntil=10000
if state['iteration']==0:
	M['proposeAfter']=pretrainUntil
	M['nextProposalTask']=0
for state['iteration'] in range(state['iteration']+1, 50001):
	i = state['iteration']

	opt.zero_grad()
	inputs, target = getBatch(args.batch_size)
	score = net.score(inputs, target).mean()
	(-score).backward()
	opt.step()

	state['training_losses'].append(-score.data[0])
	state['score'] = score.data[0]
	print("Network score %3.3f" % state['score'])

	if i>M['proposeAfter'] and i%5==0:
		print()
		print("Proposing on task: %d/%d (%d instances)" % (M['nextProposalTask'], len(data), len(data[M['nextProposalTask']])))
		libraryUpdated = makeProposalOnCol(M['nextProposalTask'])
		M['nextProposalTask'] = (M['nextProposalTask']+1)%len(data)
		print()
		# if libraryUpdated:
			# for j in range(len(data)): makeProposalOnCol(j, chr(128+len(library)-1))
			# M['proposeAfter'] = i+50

	if i%20==0:
		print("Library:", ", ".join([regex.humanreadable(x['base']) for x in library]))

	if i>0 and i%200==0:
		plt.clf()
		plt.plot(range(1, i+1), state['training_losses'])
		plt.xlim(xmin=0, xmax=i)
		plt.ylim(ymin=0, ymax=10)
		plt.xlabel('iteration')
		plt.ylabel('NLL')
		plt.savefig(plotfile)
		M['optstate'] = opt.state_dict()
		print("Saving...")
		model.save(M)
	if i==pretrainUntil: model.saveIteration(M)