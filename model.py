import string
import random
from collections import Counter, namedtuple, OrderedDict
import torch
import numpy as np
import regex2 as regex

# Save/Load

def load(file):
	M = torch.load(file, map_location=lambda storage, loc: storage.cuda() if torch.cuda.is_available() else storage)
	if 'data_file' not in M: M['data_file'] = './data/csv_old.p'
	if 'model_file' not in M: M['model_file'] = file
	if 'results_file' not in M: M['results_file'] = 'results/' + M['args'].name + '.txt'
	if 'proposeAfter' not in M: M['proposeAfter'] = 10000
	if 'nextProposalTask' not in M: M['nextProposalTask'] = 0
	return M

def save(M):
	def char_map(char):
		idx=ord(char)-128
		return regex.humanreadable(M['library'][idx]['base'], char_map)
	torch.save(M, M['model_file'])
	with open(M['results_file'], "w") as text_file:
		for i in range(len(M['trace'].concepts)):
			c = M['trace'].concepts[i]
			base = regex.humanreadable(c.base, char_map=char_map)
			numTasks = len([k for k in M['trace'].task_observations if any(x.concept==c for x in M['trace'].task_observations[k])])
			text_file.write("%d (%d uses): %s\n" % (i, numTasks, base))

def saveIteration(M):
	torch.save(M, M['model_file'] + "_" + str(M['state']['iteration']))





# Model
Observation = namedtuple("Observation", ['string', 'concept', 'ancestors', 'task'])

class Concept:
	"""
	self.nObservations is key:value pairs for CRP tables: number of customers
	"""
	def __init__(self, base):
		self.id = random.randint(1, 10000000)
		print("Need to fix concept id. When copying a concept, we don't want the neural network to think it's totally new")
		self.base = base
		self.nObservations = Counter() # Counter<Observation>
		self.nObservations_total = 0
		self.string_observations = {} # dict<str,list<Observation>>
		self.descendant_observations = [] # list<Observation>

	def __hash__(self):
		return self.id

	def sample(self):
		table = np.random.choice(
			list(nObservations.keys()) + [None],
			p=[count/(1+self.nObservations_total) for count in self.nObservations.values()] + [1/(1+self.nObservations_total)])
		if table == None:
			return regex.sample(self.base)
		else:
			return table.string

	def match(self, string, obs_ancestors):
		"""
		Attempts to match concept to "string"
		Does not update observations
		Returns score, new_observations
			where new_observations is [(string, concept, ancestors), ...]
		"""
		if string in self.string_observations:
			n_customers_at_table = concept.nObservations(concept.string_observations[string][0]) #If multiple tables have same string, put in the first
			p_same_table = n_customers_at_table / (concept.nObservations_total + 1)
			return p_same_table, [(string, self, obs_ancestors)]
		else:
			p_new_table = 1 / (concept.nObservations_total + 1)
			score, new_observations = regex.match(string, concept.base, lib_score=lib_score, mode="full", obs_concept=self, obs_ancestors=obs_ancestors)
			return p_new_table+score, new_observations 

class Trace:
	def __init__(self, initialConcept, tasks):
		self.concepts = []
		self.task_observations = {} # dict<key, <list<Observation>>>
		self.task_concept = {} #dict<key, key>
		self.score = 0
		
		for i in range(len(tasks)):
			c = Counter(tasks[i])
			for string in c:
				self.addObs(string, initialConcept, task=i, count=c[string])
				self.task_concept[i] = initialConcept

	def addObs(self, string, concept, ancestors=(), task=None, count=1):
		obs = Observation(string, concept, ancestors, task)
		if concept not in self.concepts: self.concepts.append(concept)
		if string not in concept.string_observations: concept.string_observations[string] = []
		if obs not in concept.string_observations[string]: concept.string_observations[string].append(obs)
		
		concept.nObservations[obs] += count
		concept.nObservations_total += count

		for ancestor in ancestors:
			if obs not in ancestor.descendant_observations: ancestor.descendant_observations.append(obs)
		if task not in self.task_observations: self.task_observations[task] = []
		if obs not in self.task_observations[task]: self.task_observations[task].append(obs)

	def removeObs(self, obs):
		concept.nObservations_total -= nObservations[obs]
		concept.nObservations.remove(obs)
		concept.string_observations[obs.string].remove(obs)
		if not concept.string_observations[obs.string]: 
			del concept.string_observations[obs.string]

		if obs.concept.nObservations_total == 0:
			self.concepts.remove(concept)
		for ancestor in obs.ancestors:
			ancestor.descendant_observations.remove(obs)
		self.task_observations[obs.task].remove(obs)
		self.score = None

	def clearTask(self, task):
		for obs in self.task_observations[task]:
			self.removeObs(obs)
			del self.task_concept[task]

	def setTaskConcept(self, task, concept):
		self.task_concept[task] = concept

	def fork(self):
		start = time.time()
		forked = copy.deepcopy(self)
		print("fork took", time.time()-start, "seconds")
		return fork

	def observeConcept(self, string, concept, obs_ancestors=(), obs_task=None):
		"""
		Attempts to draw string from concept
		- If valid, update trace and score
		- If invalid, set score = -inf
		"""
		score, new_observations = concept.match(string, obs_ancestors)
		self.score += score
		for string, concept, ancestors in new_observations:
			self.addObs(string, concept, ancestors, obs_task)

	def makeTaskProposal(self, task, from_concept, to_concept):
		trace = self.fork()
		trace.clearTask(task)
		trace.score = 0
		from_trace = trace
		to_trace = trace.fork()

		from_trace.task_concept[task] = from_concept
		to_trace.task_concept[task] = to_concept
		for string in data[task]:
			from_trace.observe(string, from_concept, obs_task=task)
			to_trace.observe(string, to_concept, obs_task=task)


		if to_trace.score > from_trace.score:
			return to_trace
		else:
			return self

	def scoreProposal(self, examples, concept):
		trace = self.fork()
		trace.score = 0
		for string in examples:
			trace.observe(string, concept)
		return score


def prior(r):
	return -math.log(40)*len(r)



# def scoreExistingConcept(trace, strings, concept, obs_task):
# 	trace = trace.fork()
# 	score = 0
# 	observations = []
# 	for string in strings:
# 		string_score, string_observations = CRPscore(trace, string, concept, obs_ancestors=[], obs_task=obs_task)
# 		if string_score == float('-inf'): return float('-inf'), trace
# 		score += string_score

# 	return score, trace

