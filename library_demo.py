import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from collections import Counter
import math
import pickle
import random

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from robustfill import RobustFill, tensorToStrings, stringsToTensor
import regex
import util
import model


#Model
print("Loading models")
# M = model.load('./models/library_dream_anaconda-project_4749192711.pt')
M = model.load('./models/library_dream_anaconda-project_7626798228.pt')
if torch.cuda.is_available():
	torch.set_default_tensor_type('torch.cuda.FloatTensor')
	M['net'].cuda()

# Data
data = pickle.load(open(M['data_file'], 'rb'))
v_input = 128
v_output = 128 + len(data)

def lib_sample(lib_char):
	idx = ord(lib_char)-128
	return random.choice(data[idx]['data'])
def char_map(lib_char):
	idx = ord(lib_char)-128
	return data[idx]['name']
def lib_score(lib_char, s):
	idx = ord(lib_char)-128
	prob = data[idx]['data'].count(s) / len(data[idx]['data'])
	return float("-inf") if prob==0 else math.log(prob)

while True:
	print("-"*20)
	print()
	print("Please enter examples:")
	examples = []
	nextInput = True
	while nextInput:
		s = input()
		if s=="":
			nextInput=False
		else:
			examples.append(s)
	
	net = M['net']
	args = M['args']
	# print("model:", args, "iteration:", model['state']['iteration'], "score:", model['state']['score'])

	outputs = []
	for i in range(10):
		inputs = [stringsToTensor(net.v_input, [example]*500) for example in examples]
		outputs.extend(tensorToStrings(net.v_output, net.sample(inputs)))
		# outputs = tensorToStrings(net.v_output, net.sample(inputs))

	d = Counter(outputs)
	outputs = sorted(d, key=d.get, reverse=True)
	if len(outputs)>30: outputs = outputs[:30]

	# for r in outputs:
	# 	print(str(d.get(r)).ljust(5), regex.humanreadable(r, char_map).ljust(20), regex.sample(r, lib_sample))
	if args.mode=="synthesis":
		outputs = list(set(outputs))
		def getScoreOnCorrect(r):
			try:
				scores = [regex.match(s, r, lib_score) for s in examples]
				return sum([x for x in scores if x>float("-inf")]) - math.log(40)*len(r)
			except regex.RegexException:
				return float('-inf')
		def getNumAccept(r):
			try:
				scores = [regex.match(s, r, lib_score) for s in examples]
				return len([x for x in scores if x>float("-inf")])
			except regex.RegexException as e:
				return 0
		def logsumexp(l): #t: List
			m = max(l)
			return math.log(sum([math.exp(x-m) for x in l])) + m
		def getNoisyScore(r):
			try:
				noisyScores = [
					logsumexp([
						math.log(0.99) + regex.match(s, r, lib_score),
						math.log(0.01) + regex.match(s, regex._['.'] + regex._['*'], lib_score)
					]) for s in examples
				]
				return sum(noisyScores) - math.log(40)*len(r)
			except regex.RegexException:
				return float('-inf')

		# d = Counter(outputs)
		# outputs.sort(key=lambda r: (-getNoisyScore(r), -d.get(r)))
		# outputs = outputs[:10]
		# for r in outputs:
		# 	s = " ".join([regex.sample(r, lib_sample).ljust(25) for i in range(5)])
		# 	print(("%3.3f" % getNoisyScore(r)).ljust(10) + regex.humanreadable(r, char_map).ljust(60) + s)
				

		d = Counter(outputs)
		outputs.sort(key=lambda r: (-getNumAccept(r), -getScoreOnCorrect(r), -d.get(r)))

		if getNumAccept(outputs[0]) == len(examples):
			outputs = [r for r in outputs if getNumAccept(r)==len(examples)]
			for r in outputs:
				s = " ".join([regex.sample(r, lib_sample).ljust(25) for i in range(5)])
				print(("%3.3f" % getScoreOnCorrect(r)).ljust(10) + regex.humanreadable(r, char_map).ljust(60) + s)
		else:
			print("No solutions found. Guesses:\n")
			outputs = outputs[:10]
			for r in outputs:
				s = " ".join([regex.sample(r, lib_sample).ljust(25) for i in range(5)])
				print(("(%d/%d)" % (getNumAccept(r), len(examples))).ljust(10) + regex.humanreadable(r, char_map).ljust(60) + s)
			
	# if args.mode=="induction":
	# 	d = Counter(outputs)
	# 	outputs = sorted(d, key=d.get, reverse=True)
	# 	if len(outputs)>10: outputs = outputs[:10]

	# 	for s in outputs: print(str(d.get(s)).ljust(5), s)
	print()