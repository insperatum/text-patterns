import torch
import shutil
from collections import Counter
import numpy as np
import pickle
import util

import render

# Save/Load
def load(file, cuda=False):
	M = torch.load(file, map_location=lambda storage, loc: storage.cuda() if cuda else storage)
	M['model_file'] = file
	return M

def save(M, append_str=""):
	torch.save(M, M['model_file'] + append_str)
	
	if 'trace' in M:
		temp_file = M['results_file']+"temp"
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
					if t[0].concept in M['trace'].baseConcepts:
						text_file.write(t[0].concept.str(M['trace']) + "\n")
					else:
						text_file.write("???\n")
					text_file.write(", ".join(list(set([x.value for x in t]))[:30]) + "\n\n")
		shutil.copy(temp_file, M['results_file']+append_str + "_results.txt")

		render.saveConcepts(M, M['results_file'] + "_concepts")
		render.saveTrainingError(M, M['results_file'] + "_plot.png")


def saveIteration(M):
	torch.save(M, M['model_file'] + "_" + str(M['state']['iteration']))

def saveCheckpoint(M):
	torch.save(M, M['model_file'] + "_task" + str(M['state']['current_task']))
	render.saveConcepts(M, M['results_file'] + "_concepts" + "_task" + str(M['state']['current_task']))

def loadData(file, n_examples, n_tasks):
	rand = np.random.RandomState()
	rand.seed(0)
	
	max_length = 15

	all_tasks = []
	for x in pickle.load(open(file, 'rb')):
		elems_filtered = [elem for elem in x['data'] if len(elem)<max_length]
		if len(elems_filtered)==0: continue

		task = rand.choice(elems_filtered, size=min(len(elems_filtered), n_examples), replace=False).tolist()
		all_tasks.append(task)

	data = []
	all_tasks = sorted(all_tasks, key=lambda examples: (max(len(x) for x in examples), -util.entropy(examples)))
	tasks_unique = []

	for task in all_tasks:
		unique = set(task)
		if not any(len(unique ^ x) / len(unique) < 0.5 for x in tasks_unique): #No two tasks should have mostly the same unique elements
			data.append(task)
			tasks_unique.append(unique)
	
	data_by_max_length = [[examples for examples in data if max(len(x) for x in examples)==i] for i in range(max_length)]

	for i in range(max_length):
		rand.shuffle(data_by_max_length[i])
		data_by_max_length[i] = data_by_max_length[i][:n_tasks]
		data_by_max_length[i] = sorted(data_by_max_length[i], key=lambda examples: -util.entropy(examples))

	data = [x for examples in data_by_max_length for x in examples]
	# rand.shuffle(data)
	# if args.n_tasks is not None:
	# 	data = data[args.skip_tasks:args.n_tasks + args.skip_tasks]

	# data = sorted(data, key=lambda examples: (max(len(x) for x in examples), -len(set(examples))))

	with open("data/data_filtered.pt", 'wb') as f:
		pickle.dump(data, f)

	return data