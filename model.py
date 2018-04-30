import math
import numpy as np
from scipy.stats import geom

import pregex as pre
from trace import RegexWrapper

CONCEPT = "CONCEPT"
maxDepth=2

class RegexModel:
	def __init__(self, character_classes, alpha, geom_p, pyconcept_alpha, pyconcept_d, pConcept=0.2):
		self.pConcept = pConcept
		self.character_classes = character_classes #= [pre.dot, pre.d, pre.s, pre.w, pre.l, pre.u]
		self.alpha = alpha
		self.geom_p = geom_p
		self.pyconcept_alpha = pyconcept_alpha
		self.pyconcept_d = pyconcept_d

		self.p_regex_no_concepts = {
			pre.String: 0.5,
			pre.Concat: 0.1,
			pre.Alt: 0.1,
			pre.KleeneStar: 0.2/3, pre.Plus: 0.2/3, pre.Maybe: 0.2/3,
			CONCEPT: 0
			#Doesn't include CONCEPT
		}
		for x in self.character_classes: self.p_regex_no_concepts[x] = 0.1 / len(self.character_classes)

		self.p_regex = {**{k: p*(1-self.pConcept) for k,p in self.p_regex_no_concepts.items()}, CONCEPT: pConcept}

		valid_no_recursion = [pre.String, CONCEPT] + self.character_classes
		self.p_regex_no_recursion = \
			{k: self.p_regex[k] / sum(self.p_regex[k] for k in valid_no_recursion) if k in valid_no_recursion else 0 
			for k in self.p_regex}

		valid_no_concepts_no_recursion = [pre.String] + self.character_classes
		self.p_regex_no_concepts_no_recursion = \
			{k: self.p_regex[k] / sum(self.p_regex[k] for k in valid_no_concepts_no_recursion) if k in valid_no_concepts_no_recursion else 0
			for k in self.p_regex}

		self.logp_regex_no_concepts = {k: math.log(p) if p>0 else float("-inf") for k,p in self.p_regex_no_concepts.items()}
		self.logp_regex = {k: math.log(p) if p>0 else float("-inf") for k,p in self.p_regex.items()}
		self.logp_regex_no_recursion = {k: math.log(p) if p>0 else float("-inf") for k,p in self.p_regex_no_recursion.items()}
		self.logp_regex_no_concepts_no_recursion = {k: math.log(p) if p>0 else float("-inf") for k,p in self.p_regex_no_concepts_no_recursion.items()}

	def sampleregex(self, trace, depth=0, conceptDist="default"):
		"""
		conceptDist: 'default' assumes base concept probabilities as defined in trace
					 'uniform' assumes uniform distribution over base concepts
		"""
		if depth==0:
			p_regex = self.p_regex_no_concepts
		elif depth==maxDepth:
			p_regex = self.p_regex_no_recursion if trace.baseConcepts else self.p_regex_no_concepts_no_recursion
		else:
			p_regex = self.p_regex if trace.baseConcepts else self.p_regex_no_concepts
		
		items = list(p_regex.items())
		idx = np.random.choice(range(len(items)), p=[p for k,p in items])
		R, p = items[idx]
			
		if R == pre.String:
			s = pre.Plus(pre.dot, p=0.3).sample()
			return R(s)
		elif R in self.character_classes:
			return R
		elif R in [pre.Concat, pre.Alt]:
			n = geom.rvs(0.8, loc=1)
			values = [self.sampleregex(trace, depth+1) for i in range(n)]
			return R(values)
		elif R in [pre.KleeneStar, pre.Plus, pre.Maybe]:
			return R(self.sampleregex(trace, depth+1))
		elif R == CONCEPT:
			if conceptDist == "default":
				return RegexWrapper(np.random.choice(trace.baseConcepts, p=[math.exp(trace.logpConcept(c)) for c in trace.baseConcepts]))
			elif conceptDist == "uniform":
				return RegexWrapper(np.random.choice(trace.baseConcepts))

	def scoreregex(self, r, trace, depth=0):
		if depth==0:
			logp_regex = self.logp_regex_no_concepts
		elif depth==maxDepth:
			logp_regex = self.logp_regex_no_recursion if trace.baseConcepts else self.logp_regex_no_concepts_no_recursion
		else:
			logp_regex = self.logp_regex if trace.baseConcepts else self.logp_regex_no_concepts

		if type(r) is RegexWrapper and r.concept in trace.baseConcepts:
			return logp_regex[CONCEPT] + trace.logpConcept(r.concept)
		elif r in self.character_classes:
			return logp_regex[r]
		else:
			R = type(r)
			p = logp_regex[R]
			if R == pre.String:
				return p + pre.Plus(pre.dot, p=0.3).match(r.arg)
			elif R == pre.Concat:
				n = len(r.values)
				return p + geom(0.8, loc=1).logpmf(n) + sum([self.scoreregex(s, trace, depth=depth+1) for s in r.values])
			elif R == pre.Alt:
				n = len(r.values)
				if all(x==r.ps[0] for x in r.ps):
					param_score = math.log(1/2)
				else:
					param_score = math.log(1/2) - (len(r.ps)+1) #~AIC
				return p + geom(0.8, loc=1).logpmf(n) + param_score + sum([self.scoreregex(s, trace, depth=depth+1) for s in r.values])
			elif R in [pre.KleeneStar, pre.Plus, pre.Maybe]:
				return p + self.scoreregex(r.val, trace, depth=depth+1)
