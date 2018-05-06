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

networkCache = {} #for a set of examples, what are 'valid' regexes, and 'all' found outputs, so far 

def getNetworkRegexes(net, current_trace, examples, maxNetworkEvals=10):
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

def getProposals(net, current_trace, examples, depth=0, modes=("regex", "crp", "regex-crp"), nProposals=10): #Includes proposals from network, and proposals on existing concepts
	assert(all(x in ["regex", "crp", "regex-crp", "regex-crp-crp"] for x in modes))
	examples = tuple(sorted(examples))
	isCached = examples in networkCache

	valid_proposals = []
	def addProposal(trace, concept):
		p = evalProposal(Proposal(depth, examples, trace, concept, None, None, None), examples)
		if p.valid: valid_proposals.append(p)

	addProposal(*current_trace.addregex(pre.String(examples[0]) if len(set(examples))==1 else pre.Alt([pre.String(x) for x in set(examples)]))) #Exactly the examples

	for c in current_trace.baseConcepts:
		addProposal(current_trace.fork(), c)
		if "crp" in modes:
			t,c = current_trace.addPY(c)
			addProposal(t, c)
	
	if net is not None:	
		n_basic_proposals = len(valid_proposals)
		for r in getNetworkRegexes(net, current_trace, examples):
			if any(x in modes for x in ("regex", "regex-crp", "regex-crp-crp")):
				t,c = current_trace.addregex(r)
				if "regex" in modes: addProposal(t, c)
				if any(x in modes for x in ("regex-crp", "regex-crp-crp")):
					t,c = t.addPY(c)
					if "regex-crp" in modes: addProposal(t, c)
					if "regex-crp-crp" in modes:
						t,c = t.addPY(c)
						addProposal(t, c)
			if len(valid_proposals)>=n_basic_proposals + nProposals:
				break

	valid_proposals.sort(key=lambda proposal: proposal.final_trace.score, reverse=True)
	# scores = {proposals[i]:evals[i].trace.score for i in range(len(proposals)) if evals[i].trace is not None}
	# proposals = sorted(scores.keys(), key=lambda proposal:-scores[proposal])
	proposals = valid_proposals[:nProposals]

	if not isCached: print("Proposals:  ", ", ".join(examples), "--->", ", ".join(proposal.concept.str(proposal.trace) for proposal in proposals))

	return proposals
