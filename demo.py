import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from collections import Counter
import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from robustfill import RobustFill, tensorToStrings, stringsToTensor
import regex
import util

print("Loading models")
models = [torch.load('./models/' + d.name) for d in os.scandir('models')]
if torch.cuda.is_available():
	torch.set_default_tensor_type('torch.cuda.FloatTensor')
	for m in models: m['net'].cuda()
# models = [m for m in models if m['args'].hidden_size == 250]

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
	
	for model in models:
		net = model['net']
		args = model['args']
		print("model:", args, "iteration:", model['state']['iteration'], "score:", model['state']['score'])

		inputs = [stringsToTensor(net.v_input, [example]*1000) for example in examples]
		outputs = tensorToStrings(net.v_output, net.sample(inputs))
		if args.mode=="synthesis":
			outputs = list(set(outputs))
			def getScore(r):
				try:
					return sum([regex.match(s, r) for s in examples]) - math.log(40)*len(r)
				except regex.RegexException:
					return float('-inf')

			outputs.sort(key=getScore, reverse=True)
			if len(outputs)>10: outputs = outputs[:10]
			for r in outputs:
				score = getScore(r)
				try:
					s = " ".join([regex.sample(r).ljust(25) for i in range(3)])
				except regex.RegexException:
					s = "(invalid)"

				print(("%3.3f" % score).ljust(10) + regex.humanreadable(r).ljust(25) + s)
		if args.mode=="induction":
			d = Counter(outputs)
			outputs = sorted(d, key=d.get, reverse=True)
			if len(outputs)>10: outputs = outputs[:10]
	
			for s in outputs: print(str(d.get(s)).ljust(5), s)
		print()