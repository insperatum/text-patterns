import math
import string
import random
import numpy as np
from scipy.stats import geom

import pregex as pre
from trace import RegexWrapper

CONCEPT = "CONCEPT"

pConcept = 0.2
character_classes = [pre.dot]#, pre.d, pre.s, pre.w, pre.l, pre.u]
default_p_regex = {
	pre.String: 0.5,
	# pre.dot: 0.6,#0.6/6, pre.d: 0.6/6, pre.s: 0.6/6, pre.w: 0.6/6, pre.l: 0.6/6, pre.u: 0.6/6,
	pre.Concat: 0.1,
	pre.Alt: 0.1,
	pre.KleeneStar: 0.2/3, pre.Plus: 0.2/3, pre.Maybe: 0.2/3
	#Doesn't include CONCEPT
}
for x in character_classes: default_p_regex[x] = 0.1 / len(character_classes)


maxDepth=2

def sampleregex(trace, pConcept=pConcept, depth=0):
	if not trace.baseConcepts: pConcept=0
	p_regex = {**{k: p*(1-pConcept) for k,p in default_p_regex.items()}, CONCEPT: pConcept}
	if depth==maxDepth:
		R = pre.String
	else:
		items = list(p_regex.items())
		idx = np.random.choice(range(len(items)), p=[p for k,p in items])
		R, p = items[idx]
		
	if R == pre.String:
		s = pre.Plus(pre.dot, p=0.3).sample()
		return R(s)
	elif R in character_classes:
		return R
	elif R in [pre.Concat, pre.Alt]:
		n = geom.rvs(0.8, loc=1)
		values = [sampleregex(trace, pConcept, depth+1) for i in range(n)]
		return R(values)
	elif R in [pre.KleeneStar, pre.Plus, pre.Maybe]:
		return R(sampleregex(trace, pConcept, depth+1))
	elif R == CONCEPT:
		return RegexWrapper(np.random.choice(trace.baseConcepts, p=[math.exp(trace.logpConcept(c)) for c in trace.baseConcepts]))

def scoreregex(r, trace, pConcept=pConcept, depth=0):
	p_regex = {**{k: p*(1-pConcept) for k,p in default_p_regex.items()}, CONCEPT: pConcept}
	logp_regex = {k: math.log(p) if p>0 else float("-inf") for k,p in p_regex.items()}
	if depth==maxDepth:
		if type(r) is pre.String:
			return pre.Plus(pre.dot, p=0.3).match(r.arg)
		else:
			return float("-inf")
	elif type(r) is RegexWrapper and r.concept in trace.baseConcepts:
		return logp_regex[CONCEPT] + trace.logpConcept(r.concept)
	elif r in character_classes:
		return logp_regex[r]
	else:
		R = type(r)
		p = logp_regex[R]
		if R == pre.String:
			return p + pre.Plus(pre.dot, p=0.3).match(r.arg)
		elif R in [pre.Concat, pre.Alt]:
			n = len(r.values)
			return p + geom(0.8, loc=1).logpmf(n) + sum([scoreregex(s, trace, pConcept=pConcept, depth=depth+1) for s in r.values])
		elif R in [pre.KleeneStar, pre.Plus, pre.Maybe]:
			return p + scoreregex(r.val, trace, pConcept=pConcept, depth=depth+1)
