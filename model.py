import math
import string
import random
import numpy as np
from scipy.stats import geom

import regex
import trace

CONCEPT = "CONCEPT"

pConcept = 0.1
character_classes = [regex.dot]#, regex.d, regex.s, regex.w, regex.l, regex.u]
default_pregex = {
	regex.String: 0.5,
	# regex.dot: 0.6,#0.6/6, regex.d: 0.6/6, regex.s: 0.6/6, regex.w: 0.6/6, regex.l: 0.6/6, regex.u: 0.6/6,
	regex.Concat: 0.1,
	regex.Alt: 0.1,
	regex.KleeneStar: 0.1/3, regex.Plus: 0.1/3, regex.Maybe: 0.1/3
	#Doesn't include CONCEPT
}
for x in character_classes: default_pregex[x] = 0.2 / len(character_classes)


maxDepth=2

def sampleregex(concepts, pConcept=pConcept, depth=0):
	if not concepts: pConcept=0
	pregex = {**{k: p*(1-pConcept) for k,p in default_pregex.items()}, CONCEPT: pConcept}
	if depth==maxDepth:
		R = regex.String
	else:
		items = list(pregex.items())
		idx = np.random.choice(range(len(items)), p=[p for k,p in items])
		R, p = items[idx]
		
	if R == regex.String:
		s = regex.Plus(regex.dot, p=0.3).sample()
		return R(s)
	elif R in character_classes:
		return R
	elif R in [regex.Concat, regex.Alt]:
		n = geom.rvs(0.8, loc=1)
		args = [sampleregex(concepts, pConcept, depth+1) for i in range(n)]
		return R(*args)
	elif R in [regex.KleeneStar, regex.Plus, regex.Maybe]:
		return R(sampleregex(concepts, pConcept, depth+1))
	elif R == CONCEPT:
		return trace.RegexWrapper(random.choice(concepts))

def scoreregex(r, concepts, pConcept=pConcept, depth=0):
	pregex = {**{k: p*(1-pConcept) for k,p in default_pregex.items()}, CONCEPT: pConcept}
	log_pregex = {k: math.log(p) if p>0 else float("-inf") for k,p in pregex.items()}
	if depth==maxDepth:
		if type(r) is regex.String:
			return regex.Plus(regex.dot, p=0.3).match(r.arg)
		else:
			return float("-inf")
	elif type(r) is trace.RegexWrapper and r.concept in concepts:
		return log_pregex[CONCEPT] - math.log(len(concepts))
	elif r in character_classes:
		return log_pregex[r]
	else:
		R = type(r)
		p = log_pregex[R]
		if R == regex.String:
			return p + regex.Plus(regex.dot, p=0.3).match(r.arg)
		elif R in [regex.Concat, regex.Alt]:
			n = len(r.arg)
			return p + geom(0.8, loc=1).logpmf(n) + sum([scoreregex(s, concepts, pConcept=pConcept, depth=depth+1) for s in r.arg])
		elif R in [regex.KleeneStar, regex.Plus, regex.Maybe]:
			return p + scoreregex(r.val, concepts, pConcept=pConcept, depth=depth+1)
