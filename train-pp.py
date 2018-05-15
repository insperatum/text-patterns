import torch

import random
import argparse
import string

import numpy as np

from pinn import RobustFill
import loader


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default=None)
parser.add_argument('--data_file', type=str, default="./data/csv_900.p")
parser.add_argument('--model_file', type=str, default="./results/model.pt")
parser.add_argument('--mode', type=str, default="data")
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--min_examples', type=int, default=1)
parser.add_argument('--max_examples', type=int, default=1)
parser.add_argument('--max_length', type=int, default=20) #maximum length of inputs or targets
parser.add_argument('--min_iterations', type=int, default=1000) #minimum number of training iterations before next concept

parser.add_argument('--cell_type', type=str, default="LSTM")
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--embedding_size', type=int, default=128)

parser.add_argument('--n_tasks', type=int, default=25) #Per max_length
parser.add_argument('--skip_tasks', type=int, default=0)
parser.add_argument('--n_examples', type=int, default=100)

args = parser.parse_args()

print(args)
assert(args.mode in ["data", "model"])

if args.mode=="model":
	M = loader.load(args.model_file)

iteration=0
data, group_idxs, test_data = loader.loadData(args.data_file, args.n_examples, args.n_tasks, args.max_length)
train_seen = [X[:int(len(X)/2)] for X in data]
train_unseen = [X[int(len(X)/2):] for X in data]

use_cuda = torch.cuda.is_available()

if args.file is not None:
	with open(args.file, "rb") as f:
		net, scores = torch.load(f)
else:
	net = RobustFill(input_vocabularies=[string.printable], target_vocabulary=string.printable,
               			      hidden_size=args.hidden_size, embedding_size=args.embedding_size, cell_type=args.cell_type)
	scores = []

print("Created network")
if use_cuda: net.cuda()


# ----------- Network training ------------------
# Sample
def getInstance(n_examples, eval_data=None, M=None):
	"""
	Returns a single problem instance, as input/target strings
	"""
	assert((eval_data is None) != (M is None)) #Generate either from model or from data

	while True:
		if M:
			r = M['trace'].model.sampleregex(M['trace'])
			target = r.sample(M['trace'])
			inputs = ([r.sample(M['trace']) for i in range(n_examples)],)

		if eval_data:
			eval_class = random.choice(eval_data)
			inputs = (list(np.random.choice(eval_class, size=n_examples)),) 
			target = random.choice(eval_class)

		if len(target)<args.max_length and all(len(x)<args.max_length for x in inputs[0]):
			break
	return {'inputs':inputs, 'target':target}

def getBatch(batch_size, eval_data=None, M=None):
	"""
	Create a batch of problem instances, as tensors
	"""
	n_examples = random.randint(args.min_examples, args.max_examples)
	instances = [getInstance(n_examples, eval_data=eval_data, M=M) for i in range(batch_size)]
	inputs = [x['inputs'] for x in instances]
	target = [x['target'] for x in instances]
	return inputs, target

def getClassificationBatch(batch_size, way, eval_data):
	"""
	Create a batch of classification instances, as tensors
	"""
	n_examples = random.randint(args.min_examples, args.max_examples)
	instances = [getInstance(n_examples, eval_data=eval_data, M=M) for i in range(batch_size)]
	inputs = [x['inputs'] for x in instances]
	#target = [x['target'] for x in instances]
	target = [instances[i-i%way]['target'] for i in range(len(instances))]
	return inputs, target

def networkStep():
	global iteration
	if args.mode == "data":
		inputs, target = getBatch(args.batch_size, eval_data=train_seen)
	if args.mode == "model":
		inputs, target = getBatch(args.batch_size, M=M)
	network_score = net.optimiser_step(inputs, target)

	iteration += 1
	if iteration%10==0:
		print("Iteration %d" % iteration, "| Network loss: %2.2f" % (-network_score))
		print(inputs[0], "--->", "".join(net.sample(inputs)[0]))
	return network_score

def train(iterations=20000, evalEvery=500):
	while True:
		if iteration <= iterations:
			networkStep()
		else:
			break

		if iteration==1 or iteration % evalEvery == 0:
			scores.append((iteration, evalAll()))
			with open("baselines/baseline_%s_%d_%d_%d-%d.pt" % (args.mode, args.hidden_size, args.embedding_size, args.min_examples, args.max_examples), "wb") as f:
				torch.save((net, scores), f)

def eval_ll(name, eval_data):
	print("\nCalculating", name, "error")
	scores=[]
	for i in range(100):
		if i%10==0: print(i,"/",100)
		inputs, target = getBatch(args.batch_size, eval_data=eval_data)
		score = net.score(inputs, target).mean().item()
		scores.append(score)
	print("Score:", sum(scores)/len(scores))	
	return sum(scores)/len(scores)

def eval_classification(name, way, eval_data):
	print("\nCalculating", way, "way", name, "classification")
	accuracies=[]
	for i in range(100):
		if i%10==0: print(i,"/",100)
		inputs, target = getClassificationBatch(args.batch_size - args.batch_size%way, way, eval_data)
		scores = net.score(inputs, target).reshape([-1, way])
		maxscore, maxidx = scores.max(dim=0)
		accuracy = (maxidx==0).float().mean().item()
		accuracies.append(accuracy)
	print("Accuracy:", sum(accuracies)/len(accuracies))	
	return sum(accuracies)/len(accuracies)

def evalAll():
	accuracy_seen = eval_classification("train_seen", 10, train_seen)
	accuracy_unseen = eval_classification("train_unseen", 10, train_unseen)
	accuracy_test = eval_classification("test", 10, test_data)
	print("Classification Accuracy:", "seen", accuracy_seen, "unseen", accuracy_unseen, "test", accuracy_test)

	score_seen = eval_ll("train_seen", train_seen)
	score_unseen = eval_ll("train_unseen", train_unseen)
	score_test = eval_ll("test", test_data)
	print("Scores:", "seen", score_seen, "unseen", score_unseen, "test", score_test)
	return (score_seen, score_unseen, score_test)

if __name__ == "__main__":
	if args.file is None:
		train()

