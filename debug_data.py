import argparse

import loader

parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, default="./data/csv.p")
parser.add_argument('--n_tasks', type=int, default=40) #Per max_length
parser.add_argument('--n_examples', type=int, default=500)
parser.add_argument('--max_length', type=int, default=15) #maximum length of inputs or targets
args = parser.parse_args()
print("Loading data...")
data, group_idxs, test_data = loader.loadData(args.data_file, args.n_examples, args.n_tasks, args.max_length)

print("\nTraining Data:")
for X in data:
	print(X[:5])

print("\nTest Data:")
for X in test_data:
	print(X[:5])
