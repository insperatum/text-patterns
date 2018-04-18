import torch
from torch.autograd import Variable
from collections import Counter
import math
import numpy as np

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

def getKink(list): #Given a list, find the index that best separates the list (by minimising the sum of standard deviations on either side)
	s = np.array(sorted(list))
	i = min(range(1,len(s)), key=lambda i: np.std(s[i:]) + np.std(s[:i]))
	val = np.std(s[i:]) + np.std(s[:i])
	return s[i-1], val