import math
import random
from scipy.misc import logsumexp
from scipy.stats import zipf, geom
import string
import re

_ = {
	'(': chr(0),
	')': chr(1),
	'*': chr(2),
	'+': chr(3),
	'?': chr(4),
	'|': chr(5),
	'.': chr(6),
	'\\d': chr(7),
	'\\s': chr(8),
	'\\w': chr(9),
	'\\l': chr(10),
	'\\u': chr(11),
}

whitespace = ' \t'
shorthandClasses = {
	_['.']: string.ascii_letters + string.digits + string.punctuation + whitespace,
	_['\\d']: string.digits,
	_['\\s']: whitespace,
	_['\\w']: string.ascii_letters + string.digits,
	_['\\l']: string.ascii_lowercase,
	_['\\u']: string.ascii_uppercase,
}

allchars = string.ascii_letters + string.digits + string.punctuation + whitespace + "".join(_.values())

maxParses=100

class RegexException(Exception):
    pass

def humanreadable(s, char_map=None):
	def flatten(s):
		if type(s) is list:
			return "".join([_['('] + flatten(x) + _[')'] for x in s])
		else:
			return s
	s = flatten(s)

	for char in _:
		s = s.replace(char, '\\' + char)
		s = s.replace(_[char], char)
	def readable(x):
		if ord(x)>=128:
			if char_map is None:
				return "<" + str(ord(x)-128) + ">"
			else:
				return "{" + char_map(x) + "}"
		elif x in string.printable and x not in "\n\x0b\x0c":
			return x
		else:
			return '�'
	s = ''.join(readable(x) for x in s)
	return s

def listify(p):
	p = list(p)
	def recurse(x, depth=0):
		if len(x)==0:
			return [], [], depth
		elif _['('] in x and (_[')'] not in x or x.index(_[')'])>x.index(_['('])):
			idx = x.index(_['('])
			consumed, remaining, d = recurse(x[idx+1:], depth+1)
			consumed2, remaining2, d2 = recurse(remaining, d)
			return x[:idx] + [consumed] + consumed2, remaining2, d2
		elif _[')'] in x and (_['('] not in x or x.index(_['('])>x.index(_[')'])):
			idx = x.index(_[')'])
			return x[:idx], x[idx+1:], depth-1
		else:
			return x, [], depth
	consumed, remainder, depth = recurse(p)
	if len(remainder) != 0: raise RegexException(humanreadable(p))
	if depth != 0: raise RegexException(humanreadable(p))
	return consumed


def match(s, r, lib_score=None, debug=False, mode="MAP", lib_depth=0):
	"""
	:param mode: "MAP" returns the logprop of the MAP parse, "score" returns the logprob over all parses, "full" returns both the MAP score and every library observation that went into it
	"""
	if type(s) is str: s = list(s)
	if type(r) is str: r = listify(r)
	numCalls=0
	def matchInner(s, p, partial):
		"""
		Returns list of (logp, remainder, observations) pairs
		observations is [{"ancestors":[ancestors...], "obs":obs}, ...]
		"""
		if debug: print("matchInner", s, p, partial)

		nonlocal numCalls
		numCalls += 1
		if numCalls>200: raise RegexException("Max number of calls on reached on regex:" + humanreadable(r))
		if lib_depth>=3: raise RegexException("Max recursion depth:" + humanreadable(r))

		if len(p) == 0:
			if partial==True or len(s)==0:
				return [(0, s, [])]
			else:
				return []
		elif p[0] == _['*'] or p[0] == _['+'] or p[0] == _['?'] or p[0] == _['|']:
			raise RegexException(humanreadable(r))
		elif len(p)>=2 and p[1] == _['*']:
			logprobs = []

			def inner(s, i, logp_sofar, observations_sofar):
				for logp, rem, observations in matchInner(s, p[2:], partial):
					logprobs.append((geom.logpmf(i, 0.5, loc=-1) + logp + logp_sofar, rem, observations + observations_sofar))
					if len(logprobs)>maxParses: raise RegexException(humanreadable(r))
				for logp, rem, observations in matchInner(s, [p[0]], True):
					inner(rem, i+1, logp + logp_sofar, observations + observations_sofar)
			inner(s, 0, 0, [])
			return logprobs
		elif len(p)>=2 and p[1] == _['+']:
			logprobs = []

			def inner(s, i, logp_sofar, observations_sofar):
				if i>0:
					for logp, rem, observations in matchInner(s, p[2:], partial):
						logprobs.append((geom.logpmf(i, 0.5, loc=0) + logp + logp_sofar, rem, observations + observations_sofar))
						if len(logprobs)>maxParses: raise RegexException(humanreadable(r))
				for logp, rem, observations in matchInner(s, [p[0]], True):
					inner(rem, i+1, logp + logp_sofar, observations + observations_sofar)
			inner(s, 0, 0, [])
			return logprobs
		elif len(p)>=2 and p[1] == _['?']:
			logprobs = []

			for logp, rem, observations in matchInner(s, p[2:], partial):
				logprobs.append((math.log(0.5) + logp, rem, observations))
				if len(logprobs)>maxParses: raise RegexException(humanreadable(r))
			for logp, rem, observations in matchInner(s, [p[0]], True):
				for logp2, rem2, observations2 in matchInner(rem, p[2:], partial):
					logprobs.append((math.log(0.5) + logp + logp2, rem2, observations + observations2))
					if len(logprobs)>maxParses: raise RegexException(humanreadable(r))

			return logprobs
		elif len(p)>=2 and p[1] == _['|']:
			if len(p)==2: raise RegexException(humanreadable(r))
			nOpt=1
			while len(p)>=2*(nOpt+1)-1 and p[2*(nOpt)-1]==_['|']:
				nOpt+=1
			after = p[2*(nOpt)-1:]
			if debug: print("nOpt=", nOpt, "after=", after)
			logprobs = []
			for i in range(nOpt):
				for logp, rem, observations in matchInner(s, [p[2*i]] + after, partial):
					logprobs.append((math.log(1./nOpt) + logp, rem, observations))
					if len(logprobs)>maxParses: raise RegexException(humanreadable(r))
			return logprobs
		elif type(p[0]) is str and p[0] in shorthandClasses:
			charclass = shorthandClasses[p[0]]
			if len(s)>0 and s[0] in charclass:
				return [(logp + math.log(1./len(charclass)), rem, observations) for logp, rem, observations in matchInner(s[1:], p[1:], partial)]
			else:
				return []
		elif type(p[0]) is list:
			return matchInner(s, p[0] + p[1:], partial)
		elif ord(p[0]) < 128: #Character primitive
			if len(s)>0 and p[0]==s[0]:
				return matchInner(s[1:], p[1:], partial)
			else:
				return []
		else: # Library primitive
			out = []
			for i in range(1, len(s)+1):
				lib_MAP = lib_score(p[0], "".join(s[:i]), lib_depth+1)
				score = lib_MAP["score"]
				observations = lib_MAP["observations"]
				# print("Scored " + "".join(s[:i]) + " with " + str(score))
				if score > float("-inf"):
					for logp, rem, observations in matchInner(s[i:], p[1:], partial):
						new_observations = observations + [{"ancestors":[], "obs":s[:i]}]
						for x in new_observations:
							x["ancestors"].insert(0, ord(p[0])-128)
						out.append((logp + score, rem, new_observations))
			return out

	try:
		paths = matchInner(s, r, False)
	except RecursionError:
		# print("RecursionError with ", s, p, flush=True)
		if mode == "full":
			return {"score":float("-inf"), "observations":None}
		else:
			return float("-inf")
		

	if mode=="MAP" or mode=="score":
		if len(paths)==0:
			return float("-inf")
		else:
			if mode=="score":
				return logsumexp([logp for logp, rem, observations in paths])
			elif mode=="MAP":
				MAP_path = max(paths, key=lambda x: x[0])
				return MAP_path[0]		
	elif mode=="full":
		if len(paths)==0:
			return {"score":float("-inf"), "observations":None}
		else:
			MAP_path = max(paths, key=lambda x: x[0])
			return {"score":MAP_path[0], "observations":MAP_path[2]}


def sample(r, lib_sample=None, debug=False):

	if type(r) is str: r = listify(r)
	def sampleInner(p):
		if debug: print("sampleInner", p)
		if len(p) == 0:
			return ""
		elif p[0] == _['*'] or p[0] == _['+'] or p[0] == _['?'] or p[0] == _['|']:
			raise RegexException(humanreadable(r))
		elif len(p)>=2 and p[1] == _['*']:
			i = geom.rvs(0.5, loc=-1)
			s = "".join(sampleInner(p[0]) for _ in range(i))
			return s + sampleInner(p[2:])
		elif len(p)>=2 and p[1] == _['+']:
			i = geom.rvs(0.5, loc=0)
			s = "".join(sampleInner(p[0]) for _ in range(i))
			return s + sampleInner(p[2:])
		elif len(p)>=2 and p[1] == _['?']:
			s = sampleInner(p[0]) if random.choice([True, False]) else "" 
			return s + sampleInner(p[2:])
		elif len(p)>=2 and p[1] == _['|']:
			if len(p)==2: raise RegexException(humanreadable(r))
			nOpt=1
			while len(p)>=2*(nOpt+1)-1 and p[2*(nOpt)-1]==_['|']:
				nOpt+=1
			after = p[2*(nOpt)-1:]
			i = random.randint(0, nOpt-1)
			if debug: print("nOpt=", nOpt, "after=", after)
			s = sampleInner(p[2*i])
			return s + sampleInner(after)
		elif type(p[0]) is str and p[0] in shorthandClasses:
			charclass = shorthandClasses[p[0]]
			return random.choice(charclass) + sampleInner(p[1:])
		elif type(p[0]) is list:
			return sampleInner(p[0] + p[1:])
		elif p[0] in allchars:
			return p[0] + sampleInner(p[1:])
		else:
			return lib_sample(p[0])  + sampleInner(p[1:])
	
	return sampleInner(r)


def new(lib_chars='', T='S'):
	r = random.random()
	if T=='S':
		return new(lib_chars, 'T') + (new(lib_chars, 'S') if r<0.5 else '')
	elif T=='T':
		if r<0.9:
			return new(lib_chars, 'X')
		elif r<0.95:
			return new(lib_chars, 'X') + _[random.choice('*+?')]
		else:
			return _['('] + new(lib_chars, 'X') + _['|'] + new(lib_chars, 'X') + _[')']
	elif T=='X':
		if r<0.85:
			if r<0.8 or len(lib_chars)==0:
				return random.choice(string.ascii_letters + string.digits + whitespace + string.punctuation)
			else:
				return random.choice(lib_chars)
		elif r<0.95:
			return random.choice(list(shorthandClasses.keys()))
		else:
			return _['('] + new('S') + _[')']


if __name__ == "__main__":
	for j in range(10):
		p = new()
		print("Samples for %s:" % humanreadable(p))
		for i in range(8):
			print(sample(p, debug=False))
		print()