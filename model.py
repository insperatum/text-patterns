import torch
import regex

def load(file):
	M = torch.load(file, map_location=lambda storage, loc: storage.cuda() if torch.cuda.is_available() else storage)
	if 'data_file' not in M: M['data_file'] = './data/csv_old.p'
	if 'model_file' not in M: M['model_file'] = file
	if 'results_file' not in M: M['results_file'] = 'results/' + M['args'].name + '.txt'
	if 'proposeAfter' not in M: M['proposeAfter'] = 10000
	if 'nextProposalTask' not in M: M['nextProposalTask'] = 0
	return M

def save(M):
	def char_map(char):
		idx=ord(char)-128
		return regex.humanreadable(M['library'][idx]['base'], char_map)
	torch.save(M, M['model_file'])
	with open(M['results_file'], "w") as text_file:
		for i in range(len(M['library'])):
			base = regex.humanreadable(M['library'][i]['base'], char_map=char_map)
			numUsed = len(M['library'][i]['observations'])
			text_file.write("%d (%d uses): %s\n" % (i, numUsed, base))

def saveIteration(M):
	torch.save(M, M['model_file'] + "_" + str(M['state']['iteration']))