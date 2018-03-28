import torch
import shutil

import regex

# Save/Load
def load(file):
	M = torch.load(file, map_location=lambda storage, loc: storage.cuda() if torch.cuda.is_available() else storage)
	M['model_file'] = file
	return M

def save(M, append_str=""):
	torch.save(M, M['model_file']+append_str)
	
	if 'trace' in M:
		temp_file = M['results_file']+"temp"
		with open(temp_file, "w") as text_file:
			text_file.write(str(M['args']) + "\n\n")

			sortedConcepts = sorted(M['trace'].baseConcepts, key=lambda x:-x.n_observations(M['trace']))
			for i in range(len(sortedConcepts)):
				n_observations = sortedConcepts[i].n_observations(M['trace'])
				samples = [sortedConcepts[i].sample(M['trace']) for _ in range(5)]
				string = sortedConcepts[i].str(M['trace'])
				text_file.write(str(i).ljust(4) + " (" + str(n_observations).rjust(5) + " observations): " + string.ljust(25) + " --> " + ", ".join(samples) + "\n")
			text_file.write("\n")
			
			for i in range(len(M['task_observations'])):
				t = M['task_observations'][i]
				if len(t)>0:
					if t[0].concept in M['trace'].baseConcepts:
						text_file.write((t[0].concept.str(M['trace']) + " ---> ").rjust(50))
					else:
						text_file.write("??? ---> ".rjust(50))
					text_file.write(", ".join(list(set([x.value for x in t]))[:10]) + "\n")
		shutil.copy(temp_file, M['results_file']+append_str)

def saveIteration(M):
	torch.save(M, M['model_file'] + "_" + str(M['state']['iteration']))