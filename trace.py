import copy
import math
import time
import random

from collections import namedtuple, Counter
import numpy as np

import regex
import model

class TempList():
	"""
	Immutable 
	add: O(len(temp))
	remove: O(len(temp))
	finalise: O(len(permanent) + len(temp))
	"""
	def __init__(self, init=[]):
		self.permanent = init
		self.tempAdd = []
		self.tempRemove = []

	def refresh(self):
		if len(self.tempAdd) + len(self.tempRemove) >= 1000: self._finalise()

	def add(self, value):
		new = copy.copy(self)
		new.tempAdd = self.tempAdd + [value]
		new.refresh()
		return new

	def remove(self, value):
		new = copy.copy(self)
		if value in self.tempAdd:
			new.tempAdd = copy.copy(new.tempAdd)
			new.tempAdd.remove(value)
		else:
			new.tempRemove = self.tempRemove + [value]
		new.refresh()
		return new

	def _finalise(self):
		self.permanent = self.permanent + self.tempAdd
		for r in self.tempRemove:
			self.permanent.remove(r)
		self.tempAdd = []
		self.tempRemove = []

class TempDict():
	"""
	Immutable 
	get: O(len(temp))
	add: O(len(temp))
	finalise: O(len(permanent) + len(temp))
	"""
	def __init__(self, init={}, default=0):
		self.permanent = init
		self.default = default
		self.temp = {}

	def __getitem__(self, key):
		if key in self.temp:
			return self.temp[key]
		else:
			return self.permanent.get(key, self.default)
	
	def refresh(self):
		if len(self.temp) >= 1000: self._finalise()

	def set(self, key, value):
		new = copy.copy(self)
		new.temp = copy.copy(self.temp)

		new.temp[key] = value
		new.refresh()
		return new

	def keys(self):
		return set(self.permanent.keys()).union(set(self.temp.keys()))

	def values(self):
		return [self[key] for key in self.keys()]

	def _finalise(self):
		self.permanent = {key:self[key] for key in self.keys() if self[key] != self.default}
		self.temp = {}


class Observation(namedtuple("Observation", (['concept', 'value', 'score', 'children']))):
	pass
	# def __repr__(self):
	# 	return self.value

class Concept:
	def __init__(self):
		self.id = random.getrandbits(128)
	
	def __eq__(self, other):
		return issubclass(type(other), Concept) and self.id==other.id

	def __hash__(self):
		return self.id

	def __repr__(self):
		return "(" + type(self).__name__ + "_" + str(self.id)[:5] + ")"

	def __str__(self):
		return "(" + type(self).__name__ + "_" + str(self.id)[:5] + ")"

	def n_observations(self, trace):
		raise NotImplementedError()
	
	def str(self, trace):
		return str(self)

	def createState(self):
		raise NotImplementedError()
		# return ConceptObservations(direct=AddTemp(()), descendants=AddTemp(()))

	def sample(self, trace):
		"""
		Return value
		"""
		raise NotImplementedError()

	def priorScore(self, trace):
		"""
		Return score, and list of referenced concepts
		"""
		raise NotImplementedError()

	def observe(self, trace, value):
		"""
		Return (trace|None), Observation
		"""
		raise NotImplementedError()

	def observe_partial(self, trace, value, n=1):
		"""
		Return [(trace, Observation, numCharacters), ...]
		"""
		out = []
		for i in range(len(value)+1):
			new_trace, observation = trace.observe(self, value[:i], n=n)
			if new_trace is not None:
				out.append((new_trace, observation, i))
		return out

	def unobserve(self, trace, observation):
		"""
		Return trace
		"""
		raise NotImplementedError()



class CRPConcept(Concept):
	class State(namedtuple("CRPConcept_State", ['baseConcept', 'nCustomers', 'table_nCustomers', 'value_tables', 'cache'])):
		"""
		Uses 'Observation' objects to represent tables
		:param Concept baseConcept:
		:param int nCustomers:
		:param TempDict<Observation, int> table_nCustomers: Each table is an observation, and number of customers
		:param TempDict<key, list<Observation>> value_tables: #For each value, what are the corresponding tables.
		"""
		pass

	def str(self, trace):
		state = trace.getState(self)
		# return "(" + type(self).__name__ + "_" + str(self.id)[:5] + " " + state.baseConcept.str(trace) + ")"
		return "CRP_" + str(trace.baseConcepts.index(self)) + ":<" + state.baseConcept.str(trace) + ">"

	def createState(self, baseConcept):
		return CRPConcept.State(baseConcept = baseConcept,
								nCustomers = 0,
								value_tables = TempDict(default=[]),
								table_nCustomers = TempDict(default=0),
								cache = (None, None, None) #table_nCustomers, tables, p
								)

	def priorScore(self, trace):
		score = 0
		conceptsReferenced = [trace.getState(self).baseConcept]
		return score, conceptsReferenced

	def n_observations(self, trace):
		return trace.getState(self).nCustomers

	def refresh_cache(self, trace):
		state = trace.getState(self)
		if state.cache[0] == state.table_nCustomers:
			nextTables, p = state.cache[1:]
		else:
			nextTables = list(state.table_nCustomers.keys()) + [None]
			p = [state.table_nCustomers[table]/(1+state.nCustomers) for table in nextTables] + [(1/(1+state.nCustomers))]
			state = state._replace(cache=(state.table_nCustomers, nextTables, p))
			trace._setState(self, state)
		return state, nextTables, p

	def sample(self, trace):
		state, nextTables, p = self.refresh_cache(trace)

		tableidx = np.random.choice(np.arange(len(nextTables) + 1), p=p)
		if tableidx == len(nextTables): #New table
			x = state.baseConcept.sample(trace)
		else: #Existing table
			x = nextTables[tableidx].value
		return x

	def observe(self, trace, value, n=1):
		state = trace.getState(self)
		value_tables = state.value_tables
		matching_tables = value_tables[value]

		if not matching_tables: # Need to create a new table
			trace, base_observation = trace.observe(state.baseConcept, value)
			if trace == None:
				return None, None
			table = Observation(self, value, None, (base_observation,))
			value_tables = value_tables.set(value, value_tables[value] + [table])
		else: # Use existing table
			table = next(iter(matching_tables))

		for i in range(n):
			trace.score += math.log((max(1,state.table_nCustomers[table]+i)) / (state.nCustomers + i + 1))

		newState = state._replace(
			nCustomers = state.nCustomers + n,
			value_tables = value_tables,
			table_nCustomers = state.table_nCustomers.set(table, state.table_nCustomers[table] + n),
		)
		
		trace._setState(self, newState)
		return trace, table

	def observe_partial(self, trace, value, n=1):
		"""
		Return [(trace, Observation, numCharacters), ...]
		"""
		state = trace.getState(self)
		out = []
		for i in range(len(value)+1):
			substring = value[:i]
			for table in state.value_tables[substring]:
				new_trace = trace.fork()
				for i in range(n):
					new_trace.score += math.log((max(1,state.table_nCustomers[table])+i) / (state.nCustomers + i + 1))

				newState = state._replace(
					nCustomers = state.nCustomers + n,
					table_nCustomers = state.table_nCustomers.set(table, state.table_nCustomers[table] + n)
				)
				new_trace._setState(self, newState)
				out.append((new_trace, table, len(substring)))

		for new_trace, base_observation, numCharacters in trace.observe_partial(state.baseConcept, value): #New table
			matched_value = value[:numCharacters]
			new_trace = new_trace.fork()
			table = Observation(self, matched_value, None, (base_observation,))

			for i in range(n):
				new_trace.score += math.log((max(1,state.table_nCustomers[table]+i)) / (state.nCustomers + i + 1))
			
			newState = state._replace(
				nCustomers = state.nCustomers + n,
				value_tables = state.value_tables.set(matched_value, state.value_tables[matched_value] + [table]),
				table_nCustomers = state.table_nCustomers.set(table, state.table_nCustomers[table] + n),
			)
			new_trace._setState(self, newState)
			out.append((new_trace, table, numCharacters))

		return out

	def unobserve(self, trace, observation):
		state = trace.getState(self)

		if state.table_nCustomers[observation]>1: #Take one customer off table
			trace.score -= math.log((state.table_nCustomers[observation]-1) / state.nCustomers)
			newState = state._replace(
				nCustomers = state.nCustomers - 1,
				table_nCustomers = state.table_nCustomers.set(observation, state.table_nCustomers[observation]-1),
			)
			
		else: #Delete table
			for obs in observation.children:
				trace = trace.unobserve(obs)
			trace.score -= math.log(1 / state.nCustomers)
			newState = state._replace(
				nCustomers = state.nCustomers - 1,
				value_tables = state.value_tables.set(observation.value, [x for x in state.value_tables[observation.value] if x != observation]),
				table_nCustomers = state.table_nCustomers.set(observation, state.table_nCustomers[observation] - 1),
			)

		trace._setState(self, newState)
		return trace		



regexState = namedtuple("regexState", ["trace", "observations", "n"])

class RegexConcept(Concept):
	class State(namedtuple("RegexConcept_State", ['regex', 'observations'])):
		"""
		:param regex regex:
		:param TempList<Observations> observations:
		"""
		pass
	
	def str(self, trace):
		state = trace.getState(self)
		char_map = {
			regex.dot: ".",
			regex.d: "\\d",
			regex.s: "\\s",
			regex.w: "\\w",
			regex.l: "\\l",
			regex.u: "\\u",
			regex.KleeneStar: "*",
			regex.Plus: "+",
			regex.Maybe: "?",
			regex.Alt: "|",
			regex.OPEN: "(",
			regex.CLOSE: ")"
		}
		flat = state.regex.flatten(char_map=char_map, escape_strings=True)
		inner_str = "".join(["<" + x.str(trace) + ">" if issubclass(type(x), Concept) else str(x) for x in flat])
		# return "(" + type(self).__name__ + "_" + str(self.id)[:5] + " " + inner_str + ")"
		return inner_str

	def createState(self, regex):
		return RegexConcept.State(observations = TempDict(), regex = regex)

	def priorScore(self, trace):
		state = trace.getState(self)
		score = model.scoreregex(state.regex, trace.baseConcepts)
		conceptsReferenced = [x for x in state.regex.leafNodes() if issubclass(type(x), Concept)]
		return score, conceptsReferenced

	def n_observations(self, trace):
		return sum(trace.getState(self).observations.values())

	def sample(self, trace):
		state = trace.getState(self)
		regex = state.regex
		return regex.sample(trace)

	def observe(self, trace, value, n=1):
		state = trace.getState(self)
		regex = state.regex

		initScore = trace.score
		score, S = regex.match(value, state=regexState(trace=trace, observations=(), n=n), mergeState=True)

		if score == float("-inf"):
			return None, None
		else:
			trace = S.trace.fork()
			observation = Observation(self, value, score, S.observations)
			trace.score += score * n
			newState = RegexConcept.State(
				regex = state.regex,
				observations = state.observations.set(observation, n)
			)
			trace._setState(self, newState)
			return trace, observation

	def observe_partial(self, trace, value, n=1):
		"""
		Return [(trace, Observation, numCharacters), ...]
		"""
		state = trace.getState(self)
		regex = state.regex

		out = []
		initScore = trace.score
		for numCharacters, (score, S) in regex.match(value, state=regexState(trace=trace, observations=(), n=n), mergeState=True, returnPartials=True):
			matched_value = value[:numCharacters]
			new_trace = S.trace.fork()
			observation = Observation(self, matched_value, score, S.observations)
			new_trace.score += score * n
			newState = RegexConcept.State(
				regex = state.regex,
				observations = state.observations.set(observation, n)
			)
			new_trace._setState(self, newState)
			out.append((new_trace, observation, numCharacters))
		return out

	def unobserve(self, trace, observation):
		state = trace.getState(self)
		regex = state.regex

		initScore = trace.score

		for obs in observation.children:
			trace = trace.unobserve(obs)

		trace.score = trace.score - observation.score

		newState = RegexConcept.State(
			regex = state.regex,
			observations = state.observations.set(observation, state.observations[observation]-1)
		)
		trace._setState(self, newState)
		return trace


class RegexWrapper(regex.regex):
	def __init__(self, concept):
		self.concept = concept

	def __repr__(self):
		return "(" + type(self.arg).__name__ + ")"

	def flatten(self, char_map={}, escape_strings=False):
		return [char_map.get(type(self.concept), self.concept)]

	def sample(self, regexState):
		trace = regexState
		return self.concept.sample(trace)

	def consume(self, string, regexState):
		initScore = regexState.trace.score
		for new_trace, observation, numCharacters in regexState.trace.observe_partial(self.concept, string, n=regexState.n):
			new_trace = new_trace.fork()
			score = new_trace.score - initScore
			new_regexState = regexState._replace(trace=new_trace, observations=regexState.observations + (observation,))
			yield regex.PartialMatch(numCharacters=numCharacters, score=score, reported_score=0, continuation=None, state=new_regexState)


class Trace:
	def __init__(self):
		self.score = 0
		self.state = {} #dict<Concept, State>
		self.baseConcepts = []
		self.baseConcept_nReferences = {} #dict<Concept, int> number of times concept is referenced by other concepts
		self.baseConcept_nReferences_total = 0

	def fork(self):
		fork = copy.copy(self)
		fork.state = copy.copy(self.state)
		fork.baseConcepts = copy.copy(self.baseConcepts)
		return fork

	def __repr__(self):
		return repr({"score": self.score, "state": self.state})

	def observe(self, concept, value, n=1): 
		return concept.observe(self.fork(), value, n)

	def observe_partial(self, concept, value, n=1): 
		return concept.observe_partial(self.fork(), value, n)

	def unobserve(self, observation):
		return observation.concept.unobserve(self.fork(), observation)

	def observe_all(self, concept, values, max_n_counterexamples=5):
		observations = []
		counterexamples = []
		trace = self.fork()
		counter = Counter(values)
		for value, n in counter.items():
			new_trace, new_observation = concept.observe(trace, value, n)
			if new_trace is None:
				counterexamples.append(value)
				if len(counterexamples) >= max_n_counterexamples: break
			else:
				observations.extend([new_observation]*n)
				trace = new_trace

		if len(counterexamples)>0:
			p_valid = len(observations) / (len(observations) + sum(counter[x] for x in counterexamples))
			return None, None, counterexamples, p_valid
		else:
			return trace, observations, None, None

	def unobserve_all(self, observations):
		trace = self.fork()
		for observation in observations:
			trace = observation.concept.unobserve(trace, observation)
		return trace

	def getState(self, concept):
		if concept not in self.state:
			raise Exception("concept " + str(concept) + " not in state " + str(self.state))
		state = self.state.get(concept, None)
		return state

	def _setState(self, concept, state):
		self.state[concept] = state

	def _addConcept(self, conceptClass, *args, **kwargs):
		concept = conceptClass()
		state = concept.createState(*args, **kwargs)
		self.state[concept] = state
		
		prior, conceptsReferenced = concept.priorScore(self)
		# print(concept.str(self), "has prior score", prior)
		if conceptClass is RegexConcept: conceptTypePrior = math.log(0.5)
		if conceptClass is CRPConcept: conceptTypePrior = math.log(0.001)

		self.score += conceptTypePrior + prior
		for c in conceptsReferenced: #Score each reference proportional to (number of existing references+1)
			if c in self.baseConcepts: #
				# self.score += math.log(1/len(self.baseConcepts))
				self.score += math.log(self.baseConcept_nReferences.get(c, 0)+1) - math.log(self.baseConcept_nReferences_total + len(self.baseConcepts))
				self.baseConcept_nReferences_total += 1
				self.baseConcept_nReferences[c] = self.baseConcept_nReferences.get(c, 0) + 1 

		# self.baseConcepts.append(concept)
		return concept

	def addCRPregex(self, regex):
		trace = self.fork()
		regex_concept = trace._addConcept(RegexConcept, regex)
		concept = trace._addConcept(CRPConcept, regex_concept)
		# print("New state:")
		# for k in trace.state:
		# 	print(k, trace.state[k])
		# trace.baseConcepts.append(regex_concept)
		trace.baseConcepts.append(concept)
		return trace, concept

	def addCRP(self, concept):
		trace = self.fork()
		concept = trace._addConcept(CRPConcept, concept)
		trace.baseConcepts.append(concept)
		return trace, concept

	def addregex(self, regex):
		trace = self.fork()
		concept = trace._addConcept(RegexConcept, regex)
		trace.baseConcepts.append(concept)
		return trace, concept




# ------------ Unit tests ------------

if __name__=="__main__":
	import pickle
	import os

	trace = trace()
	trace, firstName = trace.addCRPregex(regex.create("\\w+"))
	trace, lastName  = trace.addCRPregex(regex.create("\\w+"))

	regex = regex.create("f l", {"f":RegexWrapper(firstName), "l":RegexWrapper(lastName)})
	trace, fullName = trace.addCRPregex(regex)


	trace, observation1 = trace.observe(fullName, "Luke Hewitt")
	trace, observation2 = trace.observe(fullName, "Kevin Ellis")
	trace, observation3 = trace.observe(fullName, "Max Nye")
	trace, observation4 = trace.observe(fullName, "Max Siegel")
	trace, observation5 = trace.observe(fullName, "Max KW")

	with open('trace_Test.p', 'wb') as file:
		pickle.dump(trace, file)
	with open('trace_Test.p', 'rb') as file:
		loadedtrace = pickle.load(file)
		assert(set(trace.state.keys()) == set(loadedtrace.state.keys()))
		trace = loadedtrace
	os.remove("trace_Test.p")

	print("\nFull Names:")
	for i in range(10):
		print(fullName.sample(trace))
	print()

	trace = trace.unobserve(observation1)
	trace = trace.unobserve(observation2)
	trace = trace.unobserve(observation3)
	trace = trace.unobserve(observation4)
	trace = trace.unobserve(observation5)
	
	assert(abs(trace.score) < 1e-7)
	
	#--------

	trace, observations, counterexample = trace.observe_all(firstName, ["Luke", "John", "Luke", "John"])
	for observation in observations:
		trace = trace.unobserve(observation)
	assert(abs(trace.score) < 1e-7)

	#--------

	trace, word = trace.addCRPregex(regex.create("\\w+"))
	trace, sentence = trace.addCRPregex(regex.create("%( %)+\\.", {"%":RegexWrapper(word)}))
	trace, paragraph = trace.addregex(regex.create("%( %)+", {"%":RegexWrapper(sentence)}))
	trace, observation = trace.observe(paragraph, 
		"Lorem Ipsum is simply dummy text of the printing and typesetting industry. " + \
		"Lorem Ipsum has been the industry standard dummy text ever since the 1500s. " + \
		"An unknown printer took a galley of type and scrambled it to make a " + \
		"type specimen book.")
	print("\nParagraphs:")
	for i in range(10):
		print("\n" + paragraph.sample(trace))
	assert(trace is not None)	
