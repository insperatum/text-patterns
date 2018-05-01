from trace import RegexWrapper
from collections import Counter, namedtuple
import pregex as pre

import numpy as np
import util

Proposal = namedtuple("Proposal", ["depth", "examples", "trace", "concept", "final_trace", "observations", "valid"]) #start with depth=0, increase depth when triggering a new proposal
def proposal_strip(self):
	return self._replace(final_trace=None, observations=None, valid=None)
Proposal.strip = proposal_strip

def evalProposal(proposal, examples, onCounterexamples=None, doPrint=False, task_idx=None):
	assert(proposal.final_trace is None and proposal.observations is None and proposal.valid is None)

	if proposal.trace.score == float("-inf"): #Zero probability under prior
		return proposal._replace(valid=False)

	trace, observations, counterexamples, p_valid = proposal.trace.observe_all(proposal.concept, examples, task=task_idx)
	if trace is None:
		if onCounterexamples is not None:
			if doPrint: print(proposal.concept.str(proposal.trace), "failed on", counterexamples, flush=True)
			onCounterexamples(proposal, counterexamples, p_valid)
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

networkCache = {}

def getNetworkRegexes(net, current_trace, examples):
	lookup = {concept: RegexWrapper(concept) for concept in current_trace.baseConcepts}
	examples = tuple(examples)
	if examples in networkCache:
		regex_count = networkCache[examples]
	else:
		regex_count = Counter()
		for i in range(10):
			inputs = [examples] * 500
			outputs = net.sample(inputs) 
			for o in outputs:
				try:
					r = pre.create(o, lookup=lookup)
					regex_count[r] += 1
				except pre.ParseException:
					pass
		networkCache[examples] = regex_count
	return regex_count

def getProposals(net, current_trace, examples, depth=0, include_crp=True): #Includes proposals from network, and proposals on existing concepts
	examples = sorted(examples)[:10] #Hashable for cache. Up to 10 input examples
	isCached = tuple(examples) in networkCache

	if net is None:
		network_regexes = []
	else:
		regex_count = getNetworkRegexes(net, current_trace, examples)
		network_regexes = sorted(regex_count, key=regex_count.get, reverse=True)
	
	addregex = current_trace.addPYregex if include_crp else current_trace.addregex

	proposals = [Proposal(depth, tuple(examples), *addregex(r), None, None, None) for r in network_regexes] + \
		[Proposal(depth, tuple(examples), current_trace.fork(), c, None, None, None) for c in current_trace.baseConcepts] + \
		[Proposal(depth, tuple(examples), *addregex(
			pre.String(examples[0]) if len(set(examples))==1 else pre.Alt([pre.String(x) for x in set(examples)])), None, None, None)] #Exactly the examples

	proposals = [evalProposal(proposal, examples) for proposal in proposals]
	proposals = [x for x in proposals if x.valid]
	proposals.sort(key=lambda proposal: proposal.final_trace.score, reverse=True)
	# scores = {proposals[i]:evals[i].trace.score for i in range(len(proposals)) if evals[i].trace is not None}
	# proposals = sorted(scores.keys(), key=lambda proposal:-scores[proposal])
	proposals = proposals[:10]

	if not isCached: print("Proposals:  ", ", ".join(examples), "--->", ", ".join(proposal.concept.str(proposal.trace) for proposal in proposals))

#	if include_crp:
#		new_proposals = []
#		for proposal in proposals:
#			new_proposals.append(proposal)
#			if type(proposal.concept) is not PYConcept:
#				new_trace, new_concept = proposal.trace.addPY(proposal.concept)
#				new_proposals.append(Proposal(depth, tuple(examples), new_trace, new_concept, None, None, None))
#		proposals = new_proposals

	return proposals
