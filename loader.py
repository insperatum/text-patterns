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

	# Legacy
	trace = M['trace']
	trace.model.refresh()
	if len(trace.baseConcepts)>0 and trace.baseConcepts[0].id != 0:
		for i in range(len(trace.baseConcepts)):
			v0 = trace.state.get(trace.baseConcepts[i], None)
			v1 = trace.baseConcept_nReferences.get(trace.baseConcepts[i], None)
			v2 = trace.baseConcept_nTaskReferences.get(trace.baseConcepts[i], None)
			if v0 is not None: del trace.state[trace.baseConcepts[i]]
			if v1 is not None: del trace.baseConcept_nReferences[trace.baseConcepts[i]]
			if v2 is not None: del trace.baseConcept_nTaskReferences[trace.baseConcepts[i]]
			trace.baseConcepts[i].id = i
			if v0 is not None: trace.state[trace.baseConcepts[i]] = v0
			if v1 is not None: trace.baseConcept_nReferences[trace.baseConcepts[i]] = v1
			if v2 is not None: trace.baseConcept_nTaskReferences[trace.baseConcepts[i]] = v2

	if not hasattr(trace, 'nextConceptID'):
		trace.nextConceptID=len(trace.baseConcepts)

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


def saveCheckpoint(M):
	torch.save(M, M['save_to'] + "model_task" + str(M['state']['current_task']) + "_iter" + str(M['state']['iteration']) + ".pt")

def saveRender(M):
	render.saveConcepts(M, M['save_to'] + "concepts" + "_task" + str(M['state']['current_task']) + ".gv")

def loadData(file, n_examples, n_tasks, max_length):
	rand = np.random.RandomState()
	rand.seed(0)

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
	
	grouped_data = [[examples for examples in data if max(len(x) for x in examples)==i] for i in range(max_length)]
	grouped_data = [X for X in grouped_data if len(X)>0]

	for i in range(len(grouped_data)):
		rand.shuffle(grouped_data[i])
		grouped_data[i] = grouped_data[i][:n_tasks]
		grouped_data[i] = sorted(grouped_data[i], key=lambda examples: -util.entropy(examples))

	data = [x for examples in grouped_data for x in examples]
	group_idxs = list(np.cumsum([len(X) for X in grouped_data])) 
	# rand.shuffle(data)
	# if args.n_tasks is not None:
	# 	data = data[args.skip_tasks:args.n_tasks + args.skip_tasks]

	# data = sorted(data, key=lambda examples: (max(len(x) for x in examples), -len(set(examples))))

	with open("data/data_filtered.pt", 'wb') as f:
		pickle.dump(data, f)

	return data, group_idxs
