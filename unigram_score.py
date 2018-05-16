#import math
from collections import Counter
import pregex as pre

#def unigram_score(X):
#	"""
#	Given a list of strings, X, calculate the maximum log-likelihood per character for a unigram model over characters (including STOP symbol)
#	"""
#	c = Counter(x for s in X for x in s)
#	c.update("end" for s in X)
#	n = sum(c.values())
#	logp = {x:math.log(c[x]/n) for x in c}
#	return sum(c[x]*logp[x] for x in c)/n

def regex_bound(X):
	c = Counter(X)
	regexes = [pre.create(".+"), pre.create("\d+"), pre.create("\w+"), pre.create("\s"),
			   pre.create("\\u+"), pre.create("\l+")]
	regex_scores = []
	for r in regexes:
		regex_scores.append(sum(c[x] * r.match(x) for x in X)/sum(c.values()))
	
	return max(regex_scores)
