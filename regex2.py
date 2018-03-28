from scipy.stats import geom
from collections import namedtuple

import random
import string
import math

# Todo:
# parse accept character classes

PartialMatch = namedtuple("PartialMatch", ["score", "numCharacters", "continuation"])

class Regex(namedtuple("Regex", ["type", "arg"])):
	def __new__(cls, arg):
		return super().__new__(cls, cls.__name__, arg)

	def __repr__(self):
		return str("(" + type(self).__name__ + " " + repr(self.arg) + ")")

	def sample(self, sampler=None):
		"""
		:param dict<concept, ->value> sampler:
		Returns value
		"""
		raise NotImplementedError()

	def match(self, string, matcher=None):
		partials = [[] for i in range(len(string)+1)]
		partials[0] = [(self, 0)]
		# partials[num characters consumed] = [(continuation, score), ...]
		
		numCalls=0
		for i in range(len(string)):
			remainder = string[i:]
			while partials[i]:
				continuation, score = partials[i].pop()
				if continuation is None: continue
				for remainderMatch in continuation.consume(remainder, matcher=matcher):
					j = i + remainderMatch.numCharacters
					partials[j].append((remainderMatch.continuation, score + remainderMatch.score))
			#Merge to find MAP
			bestScore = {}
			for continuation, score in partials[i+1]:
				if continuation not in bestScore or score > bestScore[continuation]:
					bestScore[continuation] = score
			partials[i+1] = list(bestScore.items())
		if partials[-1]:
			best = max(partials[-1], key=lambda x: x[1])
			return best[1]
		else:
			return float("-inf")

	# def match(s, matcher=None):
	# 	"""
	# 	:param s str:
	# 	:param dict<concept, str->(score,new matcher...)> matcher:
	# 	Returns score, matcher
	# 	"""
	# 	raise NotImplementedError()

	def consume(self, s, matcher=None):
		"""
		:param s str:
		Consume at least one token of s, return the score, the number of tokens consumed, and the regex of the remainder of the regex 
		Returns generator(PartialMatch)
		"""
		raise NotImplementedError()

class CharacterClass(Regex):
	def sample(self, sampler=None):
		return random.choice(self.arg)

	def consume(self, s, matcher=None):
		if s[:1] in self.arg:
			yield PartialMatch(score=-math.log(len(self.arg)), numCharacters=1, continuation=None)

class dot(CharacterClass):
	def __init__(self):
		super().init(string.ascii_letters + string.digits + string.punctuation + ' \t')

class d(CharacterClass):
	def __init__(self):
		super().init(string.digits)

class s(CharacterClass):
	def __init__(self):
		super().init(' \t')

class w(CharacterClass):
	def __init__(self):
		super().init(string.ascii_letters + string.digits)

class l(CharacterClass):
	def __init__(self):
		super().init(string.ascii_lowercase)

class u(CharacterClass):
	def __init__(self):
		super().init(string.ascii_uppercase)

class String(Regex):
	def __str__(self):
		return self.arg
		
	def sample(self, sampler=None):
		return self.arg

	def consume(self, s, matcher=None):
		if s[:len(self.arg)]==self.arg:
			yield PartialMatch(score=0, numCharacters=len(self.arg), continuation=None)

class Concat(Regex):
	def __new__(cls, *args):
		return super().__new__(cls, args)

	def __str__(self):
		return "".join(str(value) for value in self.arg)

	def sample(self, sampler=None):
		return "".join(value.sample(sampler) for value in self.arg)

	def consume(self, s, matcher=None):
		for partialMatch in self.arg[0].consume(s, matcher=matcher):
			if partialMatch.continuation is None:
				continuation = None if len(self.arg)==1 else Concat(*self.arg[1:])
			else:
				continuation = partialMatch.continuation if len(self.arg)==0 else Concat(partialMatch.continuation, *self.arg[1:])
			yield PartialMatch(score=partialMatch.score, numCharacters=partialMatch.numCharacters, continuation=continuation)

class KleeneStar(Regex):
	@property
	def p(self):
		return 0.5

	def __str__(self):
		if (type(self.arg) is String and len(self.arg.arg)==1) or issubclass(type(self.arg), CharacterClass) or type(self.arg) in (KleeneStar, Plus, Maybe):
			return str(self.arg) + "*"
		else:
			return "(" + str(self.arg) + ")*"

	def sample(self, sampler=None):
		n = geom.rvs(self.p, loc=-1)
		return "".join(self.arg.sample(sampler) for i in range(n))

	def consume(self, s, matcher=None):
		yield PartialMatch(score=math.log(self.p), numCharacters=0, continuation=None)
		for partialMatch in self.arg.consume(s, matcher=matcher):
			if partialMatch.continuation is None:
				continuation = KleeneStar(self.arg)
			else:
				continuation = Concat(partialMatch.continuation, KleeneStar(self.arg))
			#Uses memoryless property of geometric distribution
			yield PartialMatch(score=math.log(1-self.p) + partialMatch.score, numCharacters=partialMatch.numCharacters, continuation=continuation)

class Plus(Regex):
	@property
	def p(self):
		return 0.5

	def __str__(self):
		if (type(self.arg) is String and len(self.arg.arg)==1) or issubclass(type(self.arg), CharacterClass) or type(self.arg) in (KleeneStar, Plus, Maybe):
			return str(self.arg) + "+"
		else:
			return "(" + str(self.arg) + ")+"

	def sample(self, sampler=None):
		n = geom.rvs(self.p, loc=0)
		return "".join(self.arg.sample(sampler) for i in range(n))	

	def consume(self, s, matcher=None):
		for partialMatch in self.arg.consume(s, matcher=matcher):
			if partialMatch.continuation is None:
				continuation = KleeneStar(self.arg)
			else:
				continuation = Concat(partialMatch.continuation, KleeneStar(self.arg))
			#Uses memoryless property of geometric distribution
			yield PartialMatch(score=partialMatch.score, numCharacters=partialMatch.numCharacters, continuation=continuation)	

class Maybe(Regex):
	def __str__(self):
		if (type(self.arg) is String and len(self.arg.arg)==1) or issubclass(type(self.arg), CharacterClass) or type(self.arg) in (KleeneStar, Plus, Maybe):
			return str(self.arg) + "?"
		else:
			return "(" + str(self.arg) + ")?"

	def sample(self, sampler=None):
		if random.choice([True, False]):
			return self.arg.sample(sampler)
		else:
			return ""

	def consume(self, s, matcher=None):
		yield PartialMatch(score=math.log(0.5), numCharacters=0, continuation=None)
		for partialMatch in self.arg.consume(s, matcher=matcher):
			yield PartialMatch(score=math.log(0.5)+partialMatch.score, numCharacters=partialMatch.numCharacters, continuation=partialMatch.continuation)

class Alt(Regex):
	def __new__(cls, *args):
		return super().__new__(cls, args)

	def __str__(self):
		def bracket(value):
			if (type(value) is String and len(value.arg)>1) or (type(value) is Concat):
				return "(" + str(value) + ")"
			else:
				return str(value)
				
		return "|".join(bracket(value) for value in self.arg)

	def sample(self, sampler=None):
		value = random.choice(self.arg)
		return value.sample(sampler)

	def consume(self, s, matcher=None):
		for value in self.arg:
			for partialMatch in value.consume(s, matcher=matcher):
				yield PartialMatch(score=-math.log(len(self.arg))+partialMatch.score, numCharacters=partialMatch.numCharacters, continuation=partialMatch.continuation)






def parse(s):
	def precedence(x):
		return {"*":2, "+":2, "?":2, "|":1, "(":0}.get(x, 0)

	def parseToken(s):
		if len(s)==0: raise Exception

		if s[0] == "(":
			if len(remainder)<=1: raise Exception #No lookahead
			inner_lhs, inner_remainder = parseToken(remainder[1:])
			rhs, remainder = parse(inner_lhs, inner_remainder, 0, True)
			return rhs, remainder

		elif s[0] in string.ascii_letters + string.digits + string.punctuation + ' \t':
			return String(s[0]), s[1:]

		else:
			raise Exception

	def parse(lhs, remainder, min_precedence=0, inside_brackets=False):
		if not remainder:
			if inside_brackets: raise Exception
			return lhs, remainder

		elif remainder[0]==")":
			return lhs, remainder[1:]

		elif precedence(remainder[0]) < min_precedence:
			return lhs, remainder

		elif remainder[0] not in ("*", "+", "?", "|"): #Atom
			rhs, remainder = parseToken(remainder)

			while remainder:
				rhs, remainder = parse(rhs, remainder, 0, inside_brackets)

			if type(lhs) is String and type(rhs) is String:
				return String(lhs.arg + rhs.arg), remainder
			elif type(lhs) is String and type(rhs) is Concat and type(rhs.arg[0]) is String:
				return Concat(String(lhs.arg + rhs.arg[0].arg), *rhs.arg[1:]), remainder
			elif type(rhs) is Concat:
				return Concat(lhs, *rhs.arg), remainder
			else:
				return Concat(lhs, rhs), remainder

		else:
			op, remainder = remainder[0], remainder[1:]
			if op in "*+?": 
				#Don't need to Look right
				lhs = {"*":KleeneStar, "+":Plus, "?":Maybe}[op](lhs)
				return parse(lhs, remainder, min_precedence, inside_brackets)
			elif op == "|":
				#Need to look right
				rhs, remainder = parseToken(remainder)
				while remainder and precedence(remainder[0]) >= precedence(op):
					rhs, remainder = parse(rhs, remainder, precedence(op), inside_brackets)

				if type(rhs) is Alt:
					lhs = Alt(lhs, *rhs.arg)
				else:
					lhs = Alt(lhs, rhs)
				return parse(lhs, remainder, min_precedence)

	lhs, remainder = parseToken(s)
	return parse(lhs, remainder)[0]


if __name__=="__main__":
	# Unit tests
	test_cases = [
		("foo", parse("fo"), False),
		("foo", parse("foo"), True),
		("foo", parse("fooo"), False),
		("foo", parse("fo*"), True),
		("foo", parse("fo+"), True),
		("f"+"o"*50, parse("f"+"o*"*10), True),
		("foo", parse("fo**", True))
	]
	for (string, regex, matches) in test_cases:
		print("Testing", string, regex)
		assert(matches == (regex.match(string)>float("-inf")))
