import torch

def load(file):
	M = torch.load(file, map_location=lambda storage, loc: storage.cuda() if torch.cuda.is_available() else storage)
	if 'data_file' not in M: M['data_file'] = './data/csv_old.p'
	if 'model_file' not in M: M['model_file'] = file
	return M

def save(M):
	torch.save(M, M['model_file'])