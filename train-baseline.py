import torch

import random
import argparse
import string

import numpy as np

from pinn import RobustFill
import loader


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--fork', type=str, default=None)
parser.add_argument('--data_file', type=str, default="./data/csv_900.p")
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--min_examples', type=int, default=1)
parser.add_argument('--max_examples', type=int, default=4)
parser.add_argument('--max_length', type=int, default=15) #maximum length of inputs or targets
parser.add_argument('--min_iterations', type=int, default=1000) #minimum number of training iterations before next concept

parser.add_argument('--cell_type', type=str, default="LSTM")
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--embedding_size', type=int, default=128)

parser.add_argument('--n_tasks', type=int, default=25) #Per max_length
parser.add_argument('--skip_tasks', type=int, default=0)
parser.add_argument('--n_examples', type=int, default=100)

args = parser.parse_args()

iteration=0
data, group_idxs, test_data = loader.loadData(args.data_file, args.n_examples, args.n_tasks, args.max_length)

use_cuda = torch.cuda.is_available()
net = RobustFill(input_vocabularies=[string.printable], target_vocabulary=string.printable,
							 hidden_size=args.hidden_size, embedding_size=args.embedding_size, cell_type=args.cell_type)
print("Created network")
if use_cuda: net.cuda()


# ----------- Network training ------------------
# Sample
def getInstance(n_examples):
	"""
	Returns a single problem instance, as input/target strings
	"""
	while True:
		trainclass = random.choice(data)
		inputs = (list(np.random.choice(trainclass[:int(len(trainclass)/2)], size=n_examples)),) #Only train on first half, leave out second half
		target = random.choice(trainclass)
		if len(target)<args.max_length and all(len(x)<args.max_length for x in inputs[0]):
			break
	return {'inputs':inputs, 'target':target}

def getBatch(batch_size):
	"""
	Create a batch of problem instances, as tensors
	"""
	n_examples = random.randint(args.min_examples, args.max_examples)
	instances = [getInstance(n_examples) for i in range(batch_size)]
	inputs = [x['inputs'] for x in instances]
	target = [x['target'] for x in instances]
	return inputs, target

# SGD
def networkStep():
	global iteration
	inputs, target = getBatch(args.batch_size)
	network_score = net.optimiser_step(inputs, target)

	iteration += 1
	if iteration%10==0:
		print("Iteration %d" % iteration, "| Network loss: %2.2f" % (-network_score))
		print(inputs[0], "--->", "".join(net.sample(inputs)[0]))	
	return network_score

def train(iterations=10000, saveEvery=500):
	while True:
		if iteration <= iterations:
			networkStep()
		else:
			break

		if iteration % saveEvery == 0:
			with open("baseline_%d_%d_%d-%d.pt" % (args.hidden_size, args.embedding_size, args.min_examples, args.max_examples), 'wb') as f:
				torch.save(net, f)

if __name__ == "__main__":

	print("Training...")
	train(iterations=10000)
