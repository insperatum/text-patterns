import torch
from torch.autograd import Variable

def choose(matrix, idxs):
	if type(idxs) is Variable: idxs = idxs.data
	assert(matrix.ndimension()==2)
	unrolled_idxs = torch.arange(0,matrix.size()[0]).long()*matrix.size()[1] + idxs
	return matrix.view(matrix.nelement())[unrolled_idxs]

def logsumexp(t): #t: Variable
	m, _ = t.max(0, keepdim=True)
	return (t-m).exp().sum(0).log() + m[0]
