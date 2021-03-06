import math
from collections import Counter
import pickle
import util
import shutil

import torch
import numpy as np

import render
import pregex as pre

# Save/Load
def load(file, cuda=False):
	M = torch.load(file, map_location=lambda storage, loc: storage.cuda() if cuda else storage)
	M['model_file'] = file

	# Legacy
	if 'task_concepts' not in M: M['task_concepts'] = [None for _ in range(len(M['task_observations']))]
	if not hasattr(M['trace'].model, 'pyconcept_threshold'): M['trace'].model.pyconcept_threshold=0
	return M

def save(M, append_str=""):
	torch.save(M, M['save_to'] + "model.pt")
	
	if 'trace' in M:
		temp_file = M['save_to']+"temp"
		with open(temp_file, "w") as text_file:
			text_file.write(str(M['args']) + "\n\n")

			sortedConcepts = sorted(M['trace'].baseConcepts, key=lambda x:-x.n_observations(M['trace']))
			for i in range(len(sortedConcepts)):
				n_observations = sortedConcepts[i].n_observations(M['trace'])
				samples_counter = Counter(sortedConcepts[i].sample(M['trace']) for _ in range(1000))
				string = sortedConcepts[i].str(M['trace'])
				text_file.write(str(i).ljust(4) + " (" + str(n_observations).rjust(5) + " observations): " + string + "\n")
				text_file.write(", ".join("%s:%.1f%%" % (k, 100*v/sum(samples_counter.values())) for k,v in sorted(samples_counter.items(), key=lambda x:-x[1])) + "\n\n")
			text_file.write("\n" + "-"*100 + "\n")
			
			for i in range(len(M['task_observations'])):
				t = M['task_observations'][i]
				if len(t)>0:
					text_file.write("Task %d: "%i)
					if t[0].concept in M['trace'].baseConcepts:
						text_file.write(t[0].concept.str(M['trace']) + "\n")
					else:
						text_file.write("???\n")
					text_file.write(", ".join(list(set([x.value for x in t]))[:30]) + "\n\n")
		shutil.copy(temp_file, M['save_to'] + "_results.txt")

		render.saveConcepts(M, M['save_to'] + "concepts.gv")
		render.saveTrainingError(M, M['save_to'] + "plot.png")


def saveCheckpoint(M, saveNet=True):
	if saveNet:
		torch.save(M, M['save_to'] + "model_task" + str(M['state']['current_task']) + "_iter" + str(M['state']['iteration']) + ".pt")
	else:
		net = M['net']
		M['net']=None
		torch.save(M, M['save_to'] + "model_task" + str(M['state']['current_task']) + "_iter" + str(M['state']['iteration']) + "_no_net.pt")
		M['net']=net


def saveRender(M):
	render.saveConcepts(M, M['save_to'] + "concepts" + "_task" + str(M['state']['current_task']) + ".gv")

def loadData(file, n_examples, n_tasks, max_length):
	if file[-9:] == "csv_900.p":
		print("Loading csv_900.p, ignoring data params.")
		with open(file, 'rb') as f:
			return pickle.load(f)

	rand = np.random.RandomState()
	rand.seed(0)

	all_tasks = []
	for x in pickle.load(open(file, 'rb')):
		elems_filtered = [elem for elem in x['data'] if len(elem)<max_length]
		if len(elems_filtered)==0: continue

		task = rand.choice(elems_filtered, size=min(len(elems_filtered), n_examples), replace=False).tolist()
		all_tasks.append(task)

	data = []
	def lenEntropy(examples):
		return (max(len(x) for x in examples), -util.entropy(examples))

	all_tasks = sorted(all_tasks, key=lenEntropy)
	tasks_unique = []

	for task in all_tasks:
		unique = set(task)
		if not any(len(unique ^ x) / len(unique) < 0.7 for x in tasks_unique): #No two tasks should have mostly the same unique elements
			data.append(task)
			tasks_unique.append(unique)

	data = [X for X in data if not all(x == X[0] for x in X)]
	grouped_data = [[examples for examples in data if max(len(x) for x in examples[:100])==i] for i in range(max_length)]
	grouped_data = [X for X in grouped_data if len(X)>0]
	

	#pos_int_regex = pre.create("0|((1|2|3|4|5|6|7|8|9)\d*)")
	#float_regex = pre.Concat([pos_int_regex, pre.create("\.\d+")])
	num_regex = pre.create("-?0|((1|2|3|4|5|6|7|8|9)\d*)(\.\d+)?")

	test_data = []
	for i in range(len(grouped_data)):
		#rand.shuffle(grouped_data[i])
		for fil in [num_regex]:
			fil_idxs = [j for j,xs in enumerate(grouped_data[i]) if all(fil.match(x) > float("-inf") for x in Counter(xs))] #Indexes that match filter
			grouped_data[i] = [grouped_data[i][j] for j in range(len(grouped_data[i])) if j not in fil_idxs[math.ceil(0.25 * len(grouped_data[i])):]] #Keep at most 20%

		grouped_data[i].sort(key=len, reverse=True)
		test_data.extend([X for X in grouped_data[i][n_tasks:] if len(set(X))>=5])
		grouped_data[i] = grouped_data[i][:n_tasks]
		grouped_data[i].sort(key=lenEntropy)
		
	data = [x for examples in grouped_data for x in examples]
	#group_idxs = list(np.cumsum([len(X) for X in grouped_data])) 
	# rand.shuffle(data)
	# if args.n_tasks is not None:
	# 	data = data[args.skip_tasks:args.n_tasks + args.skip_tasks]

	# data = sorted(data, key=lambda examples: (max(len(x) for x in examples), -len(set(examples))))

	test_data = test_data[:-(len(test_data)%10)]
	data = data[:-((len(data) + len(test_data))%100)]
	
	data.sort(key=lenEntropy)
	test_data.sort(key=lenEntropy)

	unique_lengths = sorted(list(set([max(len(x) for x in X) for X in data])))
	group_idxs = np.cumsum([len([X for X in data if max(len(x) for x in X) == l]) for l in unique_lengths])
	return data, group_idxs, test_data
