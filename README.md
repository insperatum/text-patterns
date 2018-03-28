# text-patterns

Train with
`anaconda-project run train`

Demo with
`anaconda-project run demo`

# Todo
Training/Proposals:
- [ ] Train NN in parallel, on a separate machine?
- [ ] Repropose on existing concept 
- [ ] Given a valid inferred regex, parse examples with that regex, choose a matching subgroup, and run inference on that parse
Library:
- [ ] When removing library item, delete other redundant items that are only referenced by it
- [ ] Dirichlet distribution over concepts
Network:
- [ ] Bidirectional
- [ ] Beam search
- [ ] Check Robustfill should attend during P->FC rather than during softmax->P?
- [ ] give n_examples as input to FC
Trace:
- [ ] Clean up CRPRegexConcept. Currently concepts not in baseConcepts don't get scored in addConcept and CRPconcept has no prior on whether it has regex or CRP inside
- [X] Rich concepts get richer (DP?)
- [ ] Move all model logic out of trace.py
- [ ] Make it clearer which functions mutate, and also when a returned trace is clean/dirty
- [ ] concept.py?
- [ ] Refactor CRPConcept (shared functionality between observe and observe_partial)
- [ ] Move state (other than observations) into concept itself?
- [ ] CRPConcept: If there are existing tables with some value, sample rather than choosing the first
- [ ] ErrorConcept to allow errors?
- [ ] Replace TempDict and TempList with https://pythonhosted.org/pysistence/
Regex:
- [ ] Replace namedtuples with attrs
- [ ] use separate bracket types for each function?
- [ ] 'sample' and 'marginalise' modes
Note -- for this, KleeneStar needs to be adapted to get correct score for fo?* -> foo.
First calculate probability q=P(o?->ε), then multiply all partialmatches by 1/[1-q(1-p))]
- [ ] Should still be able to do dynamic programming to combine partialMatches that have different states, so long as the difference in state doesn't affect the continuation