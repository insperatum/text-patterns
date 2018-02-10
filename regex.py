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
	for char in _:
		s = s.replace(char, '\\' + char)
		s = s.replace(_[char], char)
	def readable(x):
		if ord(x)>=128:
			return "<" + char_map(x) + ">"
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
	if len(remainder) != 0: raise RegexException()
	if depth != 0: raise RegexException()
	return consumed


def match(s, p, lib_score=None, debug=False):
	if type(s) is str: s = list(s)
	if type(p) is str: p = listify(p)

	def matchInner(s, p, partial):
		if debug: print("matchInner", s, p, partial)
		if len(p) == 0:
			if partial==True or len(s)==0:
				return [(0, s)]
			else:
				return []
		elif p[0] == _['*'] or p[0] == _['+'] or p[0] == _['?'] or p[0] == _['|']:
			raise RegexException()
		elif len(p)>=2 and p[1] == _['*']:
			logprobs = []

			def inner(s, i, logp_sofar):
				for logp, rem in matchInner(s, p[2:], partial):
					logprobs.append((geom.logpmf(i, 0.5, loc=-1) + logp + logp_sofar, rem))
					if len(logprobs)>maxParses: raise RegexException()
				for logp, rem in matchInner(s, [p[0]], True):
					inner(rem, i+1, logp + logp_sofar)
			inner(s, 0, 0)
			return logprobs
		elif len(p)>=2 and p[1] == _['+']:
			logprobs = []

			def inner(s, i, logp_sofar):
				if i>0:
					for logp, rem in matchInner(s, p[2:], partial):
						logprobs.append((geom.logpmf(i, 0.5, loc=0) + logp + logp_sofar, rem))
						if len(logprobs)>maxParses: raise RegexException()
				for logp, rem in matchInner(s, [p[0]], True):
					inner(rem, i+1, logp + logp_sofar)
			inner(s, 0, 0)
			return logprobs
		elif len(p)>=2 and p[1] == _['?']:
			logprobs = []

			for logp, rem in matchInner(s, p[2:], partial):
				logprobs.append((math.log(0.5) + logp, rem))
				if len(logprobs)>maxParses: raise RegexException()
			for logp, rem in matchInner(s, [p[0]], True):
				for logp2, rem2 in matchInner(rem, p[2:], partial):
					logprobs.append((math.log(0.5) + logp + logp2, rem2))
					if len(logprobs)>maxParses: raise RegexException()

			return logprobs
		elif len(p)>=2 and p[1] == _['|']:
			if len(p)==2: raise RegexException()
			nOpt=1
			while len(p)>=2*(nOpt+1)-1 and p[2*(nOpt)-1]==_['|']:
				nOpt+=1
			after = p[2*(nOpt)-1:]
			if debug: print("nOpt=", nOpt, "after=", after)
			logprobs = []
			for i in range(nOpt):
				for logp, rem in matchInner(s, [p[2*i]] + after, partial):
					logprobs.append((math.log(1./nOpt) + logp, rem))
					if len(logprobs)>maxParses: raise RegexException()
			return logprobs
		elif type(p[0]) is str and p[0] in shorthandClasses:
			charclass = shorthandClasses[p[0]]
			if len(s)>0 and s[0] in charclass:
				return [(logp + math.log(1./len(charclass)), rem) for logp, rem in matchInner(s[1:], p[1:], partial)]
			else:
				return []
		elif type(p[0]) is list:
			return matchInner(s, p[0] + p[1:], partial)
		elif p[0] in allchars:
			if len(s)>0 and p[0]==s[0]:
				return matchInner(s[1:], p[1:], partial)
			else:
				return []
		else:
			out = []
			for i in range(1, len(s)+1):
				score = lib_score(p[0], "".join(s[:i]))
				# print("Scored " + "".join(s[:i]) + " with " + str(score))
				if score > float("-inf"):
					for logp, rem in matchInner(s[i:], p[1:], partial):
						out.append((logp + score, rem))
			return out
	try:
		paths = matchInner(s, p, False)
	except RecursionError:
		# print("RecursionError with ", s, p, flush=True)
		return float("-inf")
	if len(paths)==0:
		return float("-inf")
	else:
		return logsumexp([logp for logp, rem in paths])




def sample(p, lib_sample=None, debug=False):
	if type(p) is str: p = listify(p)

	def sampleInner(p):
		if debug: print("sampleInner", p)
		if len(p) == 0:
			return ""
		elif p[0] == _['*'] or p[0] == _['+'] or p[0] == _['?'] or p[0] == _['|']:
			raise RegexException()
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
			if len(p)==2: raise RegexException()
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
	
	return sampleInner(p)


def new(lib_chars='', T='S'):
	r = random.random()
	if T=='S':
		return new(lib_chars, 'T') + (new(lib_chars, 'S') if r<0.5 else '')
	elif T=='T':
		if r<0.7:
			return new(lib_chars, 'X')
		elif r<0.9:
			return new(lib_chars, 'X') + _[random.choice('*+?')]
		else:
			return new(lib_chars, 'X') + _['|'] + new(lib_chars, 'X')
	elif T=='X':
		if r<0.6:
			if r<0.4 or len(lib_chars)==0:
				return random.choice(string.ascii_letters + string.digits + whitespace + string.punctuation)
			else:
				return random.choice(lib_chars)
		elif r<0.9:
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