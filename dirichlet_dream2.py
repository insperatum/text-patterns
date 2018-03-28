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
from network import Network
import random
import argparse
import regex2 as regex
from datetime import datetime
import pickle
import model
import copy
import math
import numpy as np
import time
from collections import Counter

data_maxsize=1000000
expected_num_concepts = 100

# Todo
# Library:
#	Maintain observation COUNTS for each concept
#	When removing library item, delete other redundant items that are only referenced by it
#	Add dirichlet process
#	Allow proposals that reference items later in the library, but prevent cycles (importantly, prevent self-reference!)
#	Dirichlet distribution over concepts
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
default_name = "dirichlet"
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
parser.add_argument('--debug', dest='debug', action='store_true')
parser.set_defaults(debug=False)
args = parser.parse_args()

# Files to save:
os.makedirs("models", exist_ok = True)
os.makedirs("results", exist_ok = True)
modelfile = 'models/' + args.name + '.pt'
plotfile = 'results/' + args.name + '.png'

# Optimiser
def createOptimiser(net):
	return torch.optim.Adam(net.parameters(), lr=0.001)


# Load/create model+state
try:
	print("Loading model ", modelfile)
	M = model.load(modelfile)
	net = M['net']
	args = M['args']
	state = M['state']
	trace = M['trace']
	opt = createOptimiser(net)
	opt.load_state_dict(M['optstate'])

	rand = np.random.RandomState()
	rand.seed(0)
	data = [rand.choice(x['data'], size=min(data_maxsize, len(x['data'])), replace=False).tolist() for x in pickle.load(open(M['data_file'], 'rb'))]
	if args.debug: data = data[:100]
	rand.shuffle(data)
except FileNotFoundError:
	rand = np.random.RandomState()
	rand.seed(0)
	data = [rand.choice(x['data'], size=min(data_maxsize, len(x['data'])), replace=False).tolist() for x in pickle.load(open(args.data_file, 'rb'))]
	if args.debug: data = data[:100]
	rand.shuffle(data)

	M = {}
	initialConcept = model.Concept(regex._['.'] + regex._['+'])
	M['trace'] = trace = model.Trace(initialConcept=initialConcept, tasks=data)
	M['net'] = net = Network(input_vocabulary=[chr(i) for i in range(128)],
							 output_vocabulary=[chr(i) for i in range(128)] + trace.concepts,
							 hidden_size=args.hidden_size, embedding_size=args.embedding_size, cell_type=args.cell_type)
	M['args'] = args
	M['state'] = state = {'iteration':0, 'score':0, 'training_losses':[]}
	M['data_file'] = args.data_file
	M['model_file'] = modelfile
	M['results_file'] = 'results/' + args.name + '.txt'
	opt = createOptimiser(net)

# ------------------------------------------------------------------------------------------------

def getInstance(n_examples=1):
	"""
	Returns a single problem instance, as input/target strings
	"""
	
	# if random.random()<1/2:
	r = regex.new(trace.concepts)
	# else:
	# 	lib_idx = random.randint(0, len(library)-1)
	# 	r = chr(lib_idx+128)
	inputs = [regex.sample(r) for i in range(n_examples)]

	if args.mode == "synthesis":
		target = r
	elif args.mode == "induction":
		target = regex.sample(r)

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
	inputs = [net.inputToTensor([x['inputs'][j] for x in instances]) for j in range(n_examples)]
	target = net.outputToTensor([x['target'] for x in instances])
	return inputs, target

def refreshAdam(net):
	global opt
	new_opt = createOptimiser(net)
	for i in range(len(opt.param_groups)):
		opt.param_groups[i]['params'] = new_opt.param_groups[i]['params']

def makeProposalOnTask(task_idx):
	#Returns true if library has updated
	global net
	old_concept = trace.task_concept[task_idx]
	print(old_concept)

	unique = list(set(data[task_idx]))
	examples = list(np.random.choice(unique, size=min(5, len(unique)), replace=False)) #example[proposal][i]
	print("Examples:", examples)
	proposals = []
	for i in range(2):
		inputs = [net.inputToTensor([example]*500) for example in examples]
		proposals += model.Concept(net.tensorToOutput(net.sample(inputs)))
	proposals = list(set(proposals))[:100]
	proposals += [concept for concept in trace.concepts if concept != old_concept] #Don't propose the same thing
	
	moreExamples = list(np.random.choice(unique, size=20))
	proposals = sorted(proposals, key=lambda proposal: -(model.prior(proposal.base) + trace.scoreProposal(moreExamples, proposal)))[:1]
	print("Proposals:", proposals)

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
				print("( score_diff =", best_proposal['score_diff'], ")")
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
	return libraryUpdated

# Training
print("\nTraining...")
pretrainUntil = 50 if args.debug else 30000
saveEvery = 10 if args.debug else 300 #seconds
nextSave = time.time() + saveEvery
if state['iteration'] == 0:
	M['proposeAfter'] = pretrainUntil
	M['nextProposalTask'] = 0
for state['iteration'] in range(state['iteration']+1, 50001):
	i = state['iteration']

	opt.zero_grad()
	inputs, target = getBatch(args.batch_size)
	score = net.score(inputs, target).mean()
	(-score).backward()
	opt.step()

	state['training_losses'].append(-score.data[0])
	state['score'] = score.data[0]
	print("i=%d Network score %3.3f" % (i, state['score']))

	if i > M['proposeAfter'] and i % 5 == 0:
		print()
		print("Proposing on task: %d/%d (%d instances)" % (M['nextProposalTask'], len(data), len(data[M['nextProposalTask']])))
		libraryUpdated = makeProposalOnTask(M['nextProposalTask'])
		M['nextProposalTask'] = (M['nextProposalTask']+1)%len(data)
		print()
		# if libraryUpdated:
			# for j in range(len(data)): makeProposalOnTask(j, chr(128+len(library)-1))
			# M['proposeAfter'] = i+50

	if i % 20 == 0:
		print("Library:", ", ".join([regex.humanreadable(x.base) for x in trace.concepts]))

	if time.time()>nextSave:
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
		nextSave=time.time()+saveEvery
	if i==pretrainUntil: model.saveIteration(M)