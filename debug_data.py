import argparse

import loader

parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, default="./data/csv.p")
parser.add_argument('--n_tasks', type=int, default=40) #Per max_length
parser.add_argument('--n_examples', type=int, default=500)
parser.add_argument('--max_length', type=int, default=15) #maximum length of inputs or targets
args = parser.parse_args()
print("Loading data...")
data, group_idxs = loader.loadData(args.data_file, args.n_examples, args.n_tasks, args.max_length)

for X in data:
	print(X[:5])
