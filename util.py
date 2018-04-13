import torch
from torch.autograd import Variable
from collections import Counter
import math

def choose(matrix, idxs):
	if type(idxs) is Variable: idxs = idxs.data
	assert(matrix.ndimension()==2)
	unrolled_idxs = idxs + torch.arange(0, matrix.size(0)).type_as(idxs)*matrix.size(1)
	return matrix.view(matrix.nelement())[unrolled_idxs]

def logsumexp(t): #t: Variable
	m, _ = t.max(0, keepdim=True)
	return (t-m).exp().sum(0).log() + m[0]

def entropy(l): #Entropy of a list
	c = Counter(l)
	total = sum(c.values())
	return -sum((v/total) * (math.log(v) - math.log(total)) for v in c.values())