import copy
import math
import random

from collections import namedtuple, Counter
import numpy as np

import pregex as pre

default_string_depth = 2

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
	def __init__(self, id=None):
		self.id = id if id is not None else random.getrandbits(128)

	def __eq__(self, other):
		return issubclass(type(other), Concept) and self.id==other.id

	def __hash__(self):
		return self.id

	def __repr__(self):
		return str(self)

	def __str__(self):
		#return type(self).__name__ + str(self.id)
		return "#" + str(self.id)
	def get_observations(self, trace):
		raise NotImplementedError()

	def n_observations(self, trace):
		raise NotImplementedError()
	
	def get_name(self):
		return "#"+str(self.id)

	def str(self, trace, depth=default_string_depth, include_self=True):
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
		Return score
		"""
		raise NotImplementedError()

	def conceptsReferenced(self, trace):
		"""
		Return list of referenced concepts (including duplicates)
		"""
		raise NotImplementedError()

	def uniqueConceptsReferenced(self, trace):
		"""
		Return a list of unique referenced concepts
		"""
		output = []
		for x in self.conceptsReferenced(trace):
			if x not in output: output.append(x)
		return output

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



class PYConcept(Concept): 
	"""
	Pitman-Yor process
	"""
	class State(namedtuple("PYConcept_State", ['baseConcept', 'nCustomers', 'table_nCustomers', 'value_tables', 'cache'])):
		"""
		Uses 'Observation' objects to represent tables
		:param Concept baseConcept:
		:param int nCustomers:
		:param TempDict<Observation, int> table_nCustomers: Each table is an observation, and number of customers
		:param TempDict<key, list<Observation>> value_tables: #For each value, what are the corresponding tables.
		"""
		pass

	def str(self, trace, depth=default_string_depth, include_self=True):
		state = trace.getState(self)
		
		if depth==0:
			return self.get_name()
		elif include_self:
			return self.get_name() + "(%s)" % state.baseConcept.str(trace, depth=depth-1)
		else:
			return state.baseConcept.str(trace, depth=depth-1)

	def createState(self, baseConcept):
		return PYConcept.State(baseConcept = baseConcept,
								nCustomers = 0,
								value_tables = TempDict(default=[]),
								table_nCustomers = TempDict(default=0),
								cache = (None, None, None), #table_nCustomers, tables, p
								)

	def priorScore(self, trace):
		return 0

	def conceptsReferenced(self, trace):
		return [trace.getState(self).baseConcept]
		
	def n_observations(self, trace):
		return trace.getState(self).nCustomers

	def get_observations(self, trace):
		state = trace.getState(self)
		return [obs.value for obs in state.table_nCustomers.keys() for _ in range(state.table_nCustomers[obs])]

	def refresh_cache(self, trace):
		state = trace.getState(self)
		alpha = trace.model.pyconcept_alpha
		d = trace.model.pyconcept_d
		
		if state.cache[0] == state.table_nCustomers:
			currentTables, p = state.cache[1:]
		else:
			currentTables = list(state.table_nCustomers.keys())
			
			p_new_table = (alpha + d*len(currentTables))/(alpha+state.nCustomers)
			p_existing_table = (state.nCustomers - d*len(currentTables))/(alpha+state.nCustomers) 

			p = [p_existing_table * state.table_nCustomers[table] / state.nCustomers for table in currentTables] + [p_new_table]
			state = state._replace(cache=(state.table_nCustomers, currentTables, p))
			trace._setState(self, state)
		return state, currentTables, p

	def sample(self, trace):
		state, currentTables, p = self.refresh_cache(trace)

		tableidx = np.random.choice(np.arange(len(currentTables) + 1), p=p)
		if tableidx == len(currentTables): #New table
			x = state.baseConcept.sample(trace)
		else: #Existing table
			x = currentTables[tableidx].value
		return x

	def observe(self, trace, value, n=1):
		state = trace.getState(self)
		value_tables = state.value_tables
		matching_tables = value_tables[value]

		alpha = trace.model.pyconcept_alpha
		d = trace.model.pyconcept_d
		currentTables = list(state.table_nCustomers.keys())
		
		if not matching_tables: # Need to create a new table
			trace, base_observation = trace.observe(state.baseConcept, value)
			if trace == None:
				return None, None
			table = Observation(self, value, None, (base_observation,))
			value_tables = value_tables.set(value, value_tables[value] + [table])
		else: # Use existing table
			table = next(iter(matching_tables))

		nTables = len(currentTables)
		for i in range(n):
			if i == 0 and state.table_nCustomers[table]==0:
				p_new_table = (alpha + d*nTables)/(alpha+state.nCustomers)
				if p_new_table < trace.model.pyconcept_threshold:
					return None, None
				trace.score += math.log(p_new_table)
				nTables += 1
			else:
				p_existing_table = (state.nCustomers+i - d*nTables)/(alpha+state.nCustomers+i) 
				trace.score += math.log(p_existing_table * (state.table_nCustomers[table]+i) / (state.nCustomers+i)) #Choose an existing customer and join them

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
		alpha = trace.model.pyconcept_alpha
		d = trace.model.pyconcept_d
		currentTables = list(state.table_nCustomers.keys())
		
		out = []
		for i in range(len(value)+1):
			substring = value[:i]
			for table in state.value_tables[substring]:
				new_trace = trace.fork()
				for i in range(n):
					p_existing_table = (state.nCustomers+i - d*len(currentTables))/(alpha+state.nCustomers+i) 
					new_trace.score += math.log(p_existing_table * (state.table_nCustomers[table]+i) / (state.nCustomers+i))
					# new_trace.score += math.log((max(1,state.table_nCustomers[table])+i) / (state.nCustomers + i + 1))

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

			nTables = len(currentTables)
			failThreshold = False
			for i in range(n):
				if i == 0 and state.table_nCustomers[table]==0:
					p_new_table = (alpha + d*nTables)/(alpha+state.nCustomers)
					if p_new_table < trace.model.pyconcept_threshold: failThreshold=True
					new_trace.score += math.log(p_new_table)
					nTables += 1
				else:
					p_existing_table = (state.nCustomers+i - d*nTables)/(alpha+state.nCustomers+i) 
					new_trace.score += math.log(p_existing_table * (state.table_nCustomers[table]+i) / (state.nCustomers+i))

			if not failThreshold:
				newState = state._replace(
					nCustomers = state.nCustomers + n,
					value_tables = state.value_tables.set(matched_value, state.value_tables[matched_value] + [table]),
					table_nCustomers = state.table_nCustomers.set(table, state.table_nCustomers[table] + n),
				)
				new_trace._setState(self, newState)
				out.append((new_trace, table, numCharacters))

		return out

#	def unobserve(self, trace, observation):
#		state = trace.getState(self)
#
#		if state.table_nCustomers[observation]>1: #Take one customer off table
#			p_existing_table = (state.nCustomers-1 - d*len(currentTables))/(alpha+state.nCustomers-1) 
#			trace.score -= math.log(p_existing_table * (state.table_nCustomers[table]-1) / (state.nCustomers-1))
#			newState = state._replace(
#				nCustomers = state.nCustomers - 1,
#				table_nCustomers = state.table_nCustomers.set(observation, state.table_nCustomers[observation]-1),
#			)
#			
#		else: #Delete table
#			for obs in observation.children:
#				trace = trace.unobserve(obs)
#			p_new_table = (alpha + d*len(currentTables))/(alpha+state.nCustomers-1)
#			trace.score -= math.log(p_new_table)
#			newState = state._replace(
#				nCustomers = state.nCustomers - 1,
#				value_tables = state.value_tables.set(observation.value, [x for x in state.value_tables[observation.value] if x != observation]),
#				table_nCustomers = state.table_nCustomers.set(observation, state.table_nCustomers[observation] - 1),
#			)
#
#		trace._setState(self, newState)
#		return trace		



regexState = namedtuple("regexState", ["trace", "observations", "n"])

class RegexConcept(Concept):
	class State(namedtuple("RegexConcept_State", ['regex', 'observations'])):
		"""
		:param regex regex:
		:param TempList<Observations> observations:
		"""
		pass

	def str(self, trace, depth=default_string_depth, include_self=True):
		state = trace.getState(self)
		if depth==0:
			return self.get_name()
		elif include_self:
			return self.get_name() + "(%s)" % state.regex.str(lambda concept: concept.str(trace, depth=depth-1), escape_strings=False) 
		else:
			return state.regex.str(lambda concept: concept.str(trace, depth=depth-1), escape_strings=False) 

	def createState(self, regex):
		return RegexConcept.State(observations = TempDict(), regex = regex)

	def priorScore(self, trace):
		state = trace.getState(self)
		return trace.model.scoreregex(state.regex, trace)

	def conceptsReferenced(self, trace):
		state = trace.getState(self)
		return [x.concept for x in state.regex.leafNodes() if type(x) is RegexWrapper]

	def n_observations(self, trace):
		return sum(trace.getState(self).observations.values())

	def get_observations(self, trace):
		state = trace.getState(self)
		return [obs.value for obs in state.observations.keys() for _ in range(state.observations[obs])]

	def sample(self, trace):
		state = trace.getState(self)
		regex = state.regex
		return regex.sample(trace)

	def observe(self, trace, value, n=1):
		state = trace.getState(self)
		regex = state.regex

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

		for obs in observation.children:
			trace = trace.unobserve(obs)

		trace.score = trace.score - observation.score

		newState = RegexConcept.State(
			regex = regex,
			observations = state.observations.set(observation, state.observations[observation]-1)
		)
		trace._setState(self, newState)
		return trace


nConsumeCalls=0
class RegexWrapper(pre.Pregex):
	def __init__(self, concept):
		self.concept = concept

	def __repr__(self):
		return "(" + type(self.arg).__name__ + ")"

	def flatten(self, char_map={}, escape_strings=False):
		return [char_map.get(type(self.concept), self.concept)]

	def leafNodes(self):
		return [self]

	def sample(self, regexState):
		trace = regexState
		return self.concept.sample(trace)

	def consume(self, string, regexState):
		global nConsumeCalls
		nConsumeCalls += 1
		initScore = regexState.trace.score
		for new_trace, observation, numCharacters in regexState.trace.observe_partial(self.concept, string, n=regexState.n):
			new_trace = new_trace.fork()
			score = (new_trace.score - initScore)/regexState.n #Score per match
			new_regexState = regexState._replace(trace=new_trace, observations=regexState.observations + (observation,))
			yield pre.PartialMatch(numCharacters=numCharacters, score=score, reported_score=0, continuation=None, state=new_regexState)



class Trace:
	def __init__(self, model):
		self.score = 0
		self.state = {} #dict<Concept, State>
		self.nextConceptID = 0
		self.baseConcepts = []
		self.allConcepts = []
		self.baseConcept_nReferences = {} #dict<Concept, int> number of times concept is referenced by other concepts
		self.baseConcept_nReferences_total = 0
		self.baseConcept_nTaskReferences = {} #dict<Concept, int> number of times concept is referenced by other concepts
		self.model = model

	def fork(self):
		fork = copy.copy(self)
		fork.state = copy.copy(self.state)
		fork.baseConcepts = copy.copy(self.baseConcepts)
		fork.allConcepts = copy.copy(self.allConcepts)
		fork.baseConcept_nReferences = copy.copy(self.baseConcept_nReferences)
		fork.baseConcept_nTaskReferences = copy.copy(self.baseConcept_nTaskReferences)
		return fork

	def logpConcept(self, c):
		#Score each reference proportional to #references + alpha
		n_references = self.baseConcept_nReferences.get(c, 0)
		return math.log(n_references + self.model.alpha) - math.log(self.baseConcept_nReferences_total + len(self.baseConcepts)*self.model.alpha)

	def __repr__(self):
		return repr({"score": self.score, "state": self.state})

	def observe(self, concept, value, n=1, weight=1): 
		new_trace, new_observation = concept.observe(self.fork(), value, n)
		if weight != 1: new_trace.score = self.score + weight*(new_trace.score-self.score)
		return new_trace, new_observation

	def observe_partial(self, concept, value, n=1, weight=1): 
		partials = concept.observe_partial(self.fork(), value, n)
		if weight != 1:
			for (new_trace, new_observation, numCharacters) in partials:
				new_trace.score = self.score + weight*(new_trace.score-self.score)
		return partials

	def unobserve(self, observation):
		raise Exception("""
			Todo: Make sure to update baseConcept_nTaskReferences
		""")
		return observation.concept.unobserve(self.fork(), observation)

	def observe_all(self, concept, values, max_n_counterexamples=None, task=None, weight=1):
		observations = []
		counterexamples = []
		trace = self.fork()
		counter = Counter(values)
		for value, n in counter.items():
			new_trace, new_observation = concept.observe(trace, value, n)
			if new_trace is None:
				counterexamples.append(value)
				if max_n_counterexamples is not None and len(counterexamples) >= max_n_counterexamples: break
			else:
				observations.extend([new_observation]*n)
				trace = new_trace

		if len(counterexamples)>0:
			p_valid = len(observations) / (len(observations) + sum(counter[x] for x in counterexamples))
			return None, None, counterexamples, p_valid
		else:
			if task is not None: trace.baseConcept_nTaskReferences[concept] = trace.baseConcept_nTaskReferences.get(concept, 0) + 1
			if weight != 1: trace.score = self.score + weight*(trace.score-self.score)
			return trace, observations, None, None

	def unobserve_all(self, observations):
		trace = self.fork()
		raise Exception("""
			Todo: Make sure that we unobserve the correct number of observations
		""")
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

	def _newConcept(self, conceptClass, *args, **kwargs):
		concept = conceptClass(id=self.nextConceptID)
		self.nextConceptID += 1
		state = concept.createState(*args, **kwargs)
		self.state[concept] = state
		return concept

	def _addConcept(self, conceptClass, *args, **kwargs):
		concept = self._newConcept(conceptClass, *args, **kwargs)

		prior = concept.priorScore(self)
		conceptsReferenced = concept.conceptsReferenced(self)

		if conceptClass is RegexConcept: conceptTypePrior = math.log(0.5)
		if conceptClass is PYConcept: conceptTypePrior = math.log(0.5)

		self.score += math.log(self.model.geom_p) + conceptTypePrior + prior
		for c in conceptsReferenced:
			if c in self.baseConcepts:
				self.score += self.logpConcept(c)
				self.baseConcept_nReferences_total += 1
				self.baseConcept_nReferences[c] = self.baseConcept_nReferences.get(c, 0) + 1 

		return concept

	def addPYregex(self, regex):
		trace = self.fork()
		regex_concept = trace._addConcept(RegexConcept, regex)
		concept = trace._addConcept(PYConcept, regex_concept)
		trace.baseConcepts.append(concept)
		trace.allConcepts.append(regex_concept)
		trace.allConcepts.append(concept)
		return trace, concept

	def addPY(self, concept):
		trace = self.fork()
		concept = trace._addConcept(PYConcept, concept)
		trace.baseConcepts.append(concept)
		trace.allConcepts.append(concept)
		return trace, concept

	def addregex(self, regex):
		trace = self.fork()
		concept = trace._addConcept(RegexConcept, regex)
		trace.baseConcepts.append(concept)
		trace.allConcepts.append(concept)
		return trace, concept

	def initregex(self, regex):
		trace = self.fork()
		concept = trace._newConcept(RegexConcept, regex)
		trace.baseConcepts.append(concept)
		trace.allConcepts.append(concept)
		return trace, concept

	def getSimilarConcepts(self):
		descendants = {c:[] for c in self.baseConcepts}
		ancestors = {c:[] for c in self.baseConcepts}
		parents = {c:[] for c in self.baseConcepts}
		children = {c:[] for c in self.baseConcepts}

		def addParent(c, parent):
			ancestors[c].append(parent)
			ancestors[c].extend(ancestors[parent])
			for c2 in ancestors[c]: descendants[c2].append(c)
			parents[c].append(parent)
			children[parent].append(c)

		for c in self.baseConcepts:
			if type(c) is PYConcept:
				state = self.getState(c)
				addParent(c, state.baseConcept)
			if type(c) is RegexConcept:
				state = self.getState(c)
				if type(state.regex) is pre.Alt:
					for x in state.regex.values:
						if type(x) is RegexWrapper:
							addParent(c, x.concept)

		#return {c: descendants[c] + ancestors[c] for c in self.baseConcepts}
		return {c: parents[c] + children[c] for c in self.baseConcepts}

# ------------ Unit tests ------------

if __name__=="__main__":
	import pickle
	import os

	trace = Trace()
	trace, firstName = trace.addPYregex(pre.create("\\w+"))
	trace, lastName  = trace.addPYregex(pre.create("\\w+"))

	regex = pre.create("f l", {"f":RegexWrapper(firstName), "l":RegexWrapper(lastName)})
	trace, fullName = trace.addPYregex(regex)


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

	trace, observations, counterexample, p_valid = trace.observe_all(firstName, ["Luke", "John", "Luke", "John"])
	for observation in observations:
		trace = trace.unobserve(observation)
	assert(abs(trace.score) < 1e-7)

	#--------

	trace, word = trace.addPYregex(pre.create("\\w+"))
	trace, sentence = trace.addPYregex(pre.create("%( %)+\\.", {"%":RegexWrapper(word)}))
	trace, paragraph = trace.addregex(pre.create("%( %)+", {"%":RegexWrapper(sentence)}))
	trace, observation = trace.observe(paragraph, 
		"Lorem Ipsum is simply dummy text of the printing and typesetting industry. " + \
		"Lorem Ipsum has been the industry standard dummy text ever since the 1500s. " + \
		"An unknown printer took a galley of type and scrambled it to make a " + \
		"type specimen book.")
	print("\nParagraphs:")
	for i in range(10):
		print("\n" + paragraph.sample(trace))
	assert(trace is not None)	
