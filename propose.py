import random
import time

from trace import RegexWrapper
from collections import Counter, namedtuple
import pregex as pre

import math

import numpy as np
import util

Proposal = namedtuple("Proposal", ["depth", "net_examples", "target_examples", "init_trace", "trace", "concept", "altWith", "final_trace", "observations", "valid"]) #start with depth=0, increase depth when triggering a new proposal
def proposal_strip(self):
	return self._replace(final_trace=None, observations=None, valid=None)
Proposal.strip = proposal_strip

def evalProposal(proposal, onCounterexamples=None, doPrint=False, task_idx=None, likelihoodWeighting=1):
	assert(proposal.final_trace is None and proposal.observations is None and proposal.valid is None)
	if proposal.trace.score == float("-inf"): #Zero probability under prior
		return proposal._replace(valid=False)

	trace, observations, counterexamples, p_valid = proposal.trace.observe_all(proposal.concept, proposal.target_examples, task=task_idx, weight=likelihoodWeighting)
	if trace is None:
		if onCounterexamples is not None:
			if doPrint: print(proposal.concept.str(proposal.trace), "failed on", counterexamples, flush=True)
			onCounterexamples(proposal, counterexamples, p_valid, None)
		return proposal._replace(valid=False)
	else:
		if onCounterexamples is not None:
			scores = []

			c = Counter(proposal.target_examples)
			examples_reordered = [x for example in c for x in [example] * c[example]]
			for example in c:
				single_example_trace, observation = proposal.trace.observe(proposal.concept, example)
				scores.extend([single_example_trace.score - proposal.trace.score] * c[example])

			if min(scores) != max(scores):
				zscores = (np.array(scores)-np.mean(scores))/np.std(scores)				
				kinkval, kinkscore = util.getKink(zscores)

				outliers = [example for (example, zscore) in zip(examples_reordered, zscores) if zscore <= kinkval]
				p_valid = 1-len(outliers)/len(examples_reordered)
				onCounterexamples(proposal, outliers, p_valid, kinkscore)

		if doPrint: print(proposal.concept.str(proposal.trace), "got score: %3.3f" % trace.score, "of which observation: %3.3f" % (trace.score-proposal.trace.score), flush=True)
		return proposal._replace(final_trace=trace, observations=observations, valid=True)

networkCache = {} #for a set of examples, what are 'valid' regexes, and 'all' found outputs, so far 


def getNetworkRegexes(net, current_trace, examples, maxNetworkEvals=None):
	similarConcepts = current_trace.getSimilarConcepts()
	lookup = {concept: RegexWrapper(concept) for concept in current_trace.baseConcepts}
	def getRelatedRegexStrings(o):
		if len(o)==0:
			yield ()
		else:
			for s2 in getRelatedRegexStrings(o[1:]):
				for s1 in [o[0]] + similarConcepts.get(o[0], []):
					yield (s1,) + s2

	if maxNetworkEvals is None: maxNetworkEvals=10
	examples = tuple(sorted(examples))
	if examples in networkCache:
		for (r, count) in networkCache[examples]['valid']:
			yield (r, count)
	else:
		networkCache[examples]={'valid':[], 'all':set()}
		inputs = [[(example,) for example in examples]] * 500

		group_idx=0
		list_time_record=0
		for i in range(maxNetworkEvals):
			outputs_count=Counter(net.sample(inputs))
			for o in sorted(outputs_count, key=outputs_count.get):
				if o not in networkCache[examples]['all']:
					networkCache[examples]['all'].add(o)
					try:
						start_time=time.time()
						l=list(getRelatedRegexStrings(o))
						if time.time()-start_time > list_time_record:
							list_time_record=time.time()-start_time
							print("Making list (size", len(l), ") took", list_time_record, "seconds", pre.create(o, lookup=lookup))
						for o_related in l:
							networkCache[examples]['all'].add(o_related)
							r = pre.create(o_related, lookup=lookup)
							count = outputs_count.get(o_related)
							networkCache[examples]['valid'].append((r, count))
							yield (r, count, group_idx)
						group_idx += 1
					except pre.ParseException:
						pass

def getProposals(net, current_trace, target_examples, net_examples=None, depth=0, modes=("regex", "crp", "regex-crp"),
		nProposals=10, likelihoodWeighting=1, subsampleSize=None, altWith=None, maxNetworkEvals=None, doPrint=True): #Includes proposals from network, and proposals on existing concepts
	assert(all(x in ["regex", "crp", "regex-crp", "regex-crp-crp"] for x in modes))

	examples = net_examples if net_examples is not None else target_examples

	if subsampleSize is not None:
		counter = Counter(examples)
		min_examples, max_examples = subsampleSize
		nSubsamples = 10
		
		proposal_strings_sofar = [] #TODO: this better. Want to avoid duplicate proposals. For now, just using string representation to check...
	
		for i in range(nSubsamples):
			num_examples = random.randint(min_examples, max_examples)
			sampled_examples = list(np.random.choice(
				list(counter.keys()),
				size=min(num_examples, len(counter)),
				p=np.array(list(counter.values()))/sum(counter.values()),
				replace=True))
			for proposal in getProposals(net, current_trace, target_examples, sampled_examples, depth, modes, int(nProposals/nSubsamples), likelihoodWeighting, subsampleSize=None):
				proposal_string = proposal.concept.str(proposal.trace, depth=-1)
				if proposal_string not in proposal_strings_sofar:
					proposal_strings_sofar.append(proposal_string)
					yield proposal
			
	else:
		examples = tuple(sorted(examples))
		isCached = examples in networkCache

		cur_proposals = []
		net_proposals = []
		def addProposal(trace, concept, add_to):
			p = evalProposal(Proposal(depth, tuple(examples), tuple(examples), current_trace, trace, concept, altWith, None, None, None), likelihoodWeighting=likelihoodWeighting * len(target_examples)/len(examples))
			if p.valid: add_to.append(p)

		addProposal(*current_trace.addregex(pre.String(examples[0]) if len(set(examples))==1 else pre.Alt([pre.String(x) for x in set(examples)])), cur_proposals) #Exactly the examples

		for c in current_trace.baseConcepts:
			addProposal(current_trace.fork(), c, cur_proposals)
			if "crp" in modes:
				t,c = current_trace.addPY(c)
				addProposal(t, c, cur_proposals)

		n_cur = math.ceil(nProposals/2)
		n_net = math.floor(nProposals/2)
	#	m_net = n_net * 5

		if net is not None:	
			for (r, count, group_idx) in getNetworkRegexes(net, current_trace, examples):
				if any(x in modes for x in ("regex", "regex-crp", "regex-crp-crp")):
					t,c = current_trace.addregex(r)
					if "regex" in modes: addProposal(t, c, net_proposals)
					if any(x in modes for x in ("regex-crp", "regex-crp-crp")):
						t,c = t.addPY(c)
						if "regex-crp" in modes: addProposal(t, c, net_proposals)
						if "regex-crp-crp" in modes:
							t,c = t.addPY(c)
							addProposal(t, c, net_proposals)
				if group_idx>=n_net:
					break

		cur_proposals.sort(key=lambda proposal: proposal.final_trace.score, reverse=True)
		net_proposals.sort(key=lambda proposal: proposal.final_trace.score, reverse=True)
		
		# scores = {proposals[i]:evals[i].trace.score for i in range(len(proposals)) if evals[i].trace is not None}
		# proposals = sorted(scores.keys(), key=lambda proposal:-scores[proposal])
		proposals = cur_proposals[:n_cur] + net_proposals[:n_net]
		proposals.sort(key=lambda proposal: proposal.final_trace.score, reverse=True)

		if not isCached and doPrint: print("Proposals (ll*%2.2f): " % likelihoodWeighting , ", ".join(examples), "--->", ", ".join(
			("N:" if proposal in net_proposals else "") +
			proposal.concept.str(proposal.trace) for proposal in proposals), flush=True)

		for p in proposals:
			if tuple(sorted(examples)) == tuple(sorted(target_examples)):
				yield p._replace(target_examples=tuple(target_examples))
			else:
				yield p.strip()._replace(target_examples=tuple(target_examples))
