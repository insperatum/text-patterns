from trace import RegexWrapper
from collections import Counter, namedtuple
import pregex as pre

import time

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
		outputs_count = Counter()
		for i in range(10):
			inputs = [[(example,) for example in examples]] * 500
			outputs_count.update(net.sample(inputs))

		regex_count = Counter()
		for o in outputs_count:
			try:
				r = pre.create(o, lookup=lookup)
				regex_count[r] += outputs_count[o]
			except pre.ParseException:
				pass
		networkCache[examples] = regex_count
	return regex_count

#Regex, CRP, Regex+CRP, Regex+CRP+CRP
def getProposals(net, current_trace, examples, depth=0, modes=("regex", "crp", "regex-crp"), nProposals=10, printTimes=False): #Includes proposals from network, and proposals on existing concepts
	assert(all(x in ["regex", "crp", "regex-crp", "regex-crp-crp"] for x in modes))
	examples = sorted(examples)[:10] #Hashable for cache. Up to 10 input examples
	isCached = tuple(examples) in networkCache

	if net is None:
		network_regexes = []
	else:
		start_time = time.time()
		regex_count = getNetworkRegexes(net, current_trace, examples)
		if printTimes: print("Get network regexes: %dms" % (100*(time.time()-start_time)))
		network_regexes = sorted(regex_count, key=regex_count.get, reverse=True)

	start_time = time.time()
	proposals = [Proposal(depth, tuple(examples), *current_trace.addregex(
			pre.String(examples[0]) if len(set(examples))==1 else pre.Alt([pre.String(x) for x in set(examples)])), None, None, None)] #Exactly the examples
	for c in current_trace.baseConcepts:
		proposals.append(Proposal(depth, tuple(examples), current_trace.fork(), c, None, None, None))
		if "crp" in modes:
			t,c = current_trace.addPY(c)
			proposals.append(Proposal(depth, tuple(examples), t, c, None, None, None))

	for r in network_regexes:
		if "regex" in modes:
			t,c = current_trace.addregex(r)
			proposals.append(Proposal(depth, tuple(examples), t, c, None, None, None))
		if "regex-crp" in modes:
			t,c = current_trace.addregex(r)
			t,c = t.addPY(c)
			proposals.append(Proposal(depth, tuple(examples), t, c, None, None, None))
		if "regex-crp-crp" in modes:
			t,c = current_trace.addregex(r)
			t,c = t.addPY(c)
			t,c = t.addPY(c)
			proposals.append(Proposal(depth, tuple(examples), t, c, None, None, None))
	if printTimes: print("Make proposals: %dms" % (100*(time.time()-start_time)))

	start_time = time.time()
	if printTimes: print("Evaluating %d proposals" % len(proposals))
	proposals = [evalProposal(proposal, examples) for proposal in proposals]
	if printTimes: print("Evaluate proposals: %dms" % (100*(time.time()-start_time)))

	proposals = [x for x in proposals if x.valid]
	proposals.sort(key=lambda proposal: proposal.final_trace.score, reverse=True)
	# scores = {proposals[i]:evals[i].trace.score for i in range(len(proposals)) if evals[i].trace is not None}
	# proposals = sorted(scores.keys(), key=lambda proposal:-scores[proposal])
	proposals = proposals[:nProposals]

	if not isCached: print("Proposals:  ", ", ".join(examples), "--->", ", ".join(proposal.concept.str(proposal.trace) for proposal in proposals))

	return proposals
