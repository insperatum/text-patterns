from trace import RegexWrapper
from collections import Counter, namedtuple
import pregex as pre

import math

import numpy as np
import util

Proposal = namedtuple("Proposal", ["depth", "examples", "trace", "concept", "final_trace", "observations", "valid"]) #start with depth=0, increase depth when triggering a new proposal
def proposal_strip(self):
	return self._replace(final_trace=None, observations=None, valid=None)
Proposal.strip = proposal_strip

def evalProposal(proposal, examples, onCounterexamples=None, doPrint=False, task_idx=None, likelihoodWeighting=1):
	assert(proposal.final_trace is None and proposal.observations is None and proposal.valid is None)

	if proposal.trace.score == float("-inf"): #Zero probability under prior
		return proposal._replace(valid=False)

	trace, observations, counterexamples, p_valid = proposal.trace.observe_all(proposal.concept, examples, task=task_idx, weight=likelihoodWeighting)
	if trace is None:
		if onCounterexamples is not None:
			if doPrint: print(proposal.concept.str(proposal.trace), "failed on", counterexamples, flush=True)
			onCounterexamples(proposal, counterexamples, p_valid, None)
		return proposal._replace(valid=False)
	else:
		if onCounterexamples is not None:
			scores = []

			c = Counter(examples)
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

def getNetworkRegexes(net, current_trace, examples, maxNetworkEvals=30):
	lookup = {concept: RegexWrapper(concept) for concept in current_trace.baseConcepts}
	examples = tuple(sorted(examples))
	if examples in networkCache:
		for r in networkCache[examples]['valid']:
			yield r
	else:
		networkCache[examples]={'valid':[], 'all':set()}
		inputs = [[(example,) for example in examples]] * 500

		for i in range(maxNetworkEvals):
			outputs_count=Counter(net.sample(inputs))
			for o in sorted(outputs_count, key=outputs_count.get):
				if o not in networkCache[examples]['all']:
					networkCache[examples]['all'].add(o)
					try:
						r = pre.create(o, lookup=lookup)
						networkCache[examples]['valid'].append(r)
						yield r
					except pre.ParseException:
						pass

def getProposals(net, current_trace, examples, depth=0, modes=("regex", "crp", "regex-crp"), nProposals=10, nEffectiveExamples=None): #Includes proposals from network, and proposals on existing concepts
	assert(all(x in ["regex", "crp", "regex-crp", "regex-crp-crp"] for x in modes))
	examples = tuple(sorted(examples))
	likelihoodWeighting = nEffectiveExamples/len(examples) if nEffectiveExamples is not None else 1
	isCached = examples in networkCache

	cur_proposals = []
	net_proposals = []
	def addProposal(trace, concept, add_to):
		p = evalProposal(Proposal(depth, examples, trace, concept, None, None, None), examples, likelihoodWeighting=likelihoodWeighting)
		if p.valid: add_to.append(p)

	addProposal(*current_trace.addregex(pre.String(examples[0]) if len(set(examples))==1 else pre.Alt([pre.String(x) for x in set(examples)])), cur_proposals) #Exactly the examples

	for c in current_trace.baseConcepts:
		addProposal(current_trace.fork(), c, cur_proposals)
		if "crp" in modes:
			t,c = current_trace.addPY(c)
			addProposal(t, c, cur_proposals)

	n_cur = math.ceil(nProposals/2)
	n_net = math.floor(nProposals/2)
	m_net = n_net * 5

	if net is not None:	
		for r in getNetworkRegexes(net, current_trace, examples):
			if any(x in modes for x in ("regex", "regex-crp", "regex-crp-crp")):
				t,c = current_trace.addregex(r)
				if "regex" in modes: addProposal(t, c, net_proposals)
				if any(x in modes for x in ("regex-crp", "regex-crp-crp")):
					t,c = t.addPY(c)
					if "regex-crp" in modes: addProposal(t, c, net_proposals)
					if "regex-crp-crp" in modes:
						t,c = t.addPY(c)
						addProposal(t, c, net_proposals)
			if len(net_proposals)>=m_net:
				break

	cur_proposals.sort(key=lambda proposal: proposal.final_trace.score, reverse=True)
	net_proposals.sort(key=lambda proposal: proposal.final_trace.score, reverse=True)
	
	# scores = {proposals[i]:evals[i].trace.score for i in range(len(proposals)) if evals[i].trace is not None}
	# proposals = sorted(scores.keys(), key=lambda proposal:-scores[proposal])
	proposals = cur_proposals[:n_cur] + net_proposals[:n_net]
	proposals.sort(key=lambda proposal: proposal.final_trace.score, reverse=True)

	if not isCached: print("Proposals:  ", ", ".join(examples), "--->", ", ".join(
		("N:" if proposal in net_proposals else "") +
		proposal.concept.str(proposal.trace) for proposal in proposals))

	return proposals
