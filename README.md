# text-patterns

Train with
`anaconda-project run train`

Demo with
`anaconda-project run demo`

# Todo
IMPORTANT:
- [ ] Now regex.Alt has a continuous parameter. Need to pay for it in the score (either AIC or something variational)

Training/Proposals:
- [X] Evaluate multiple proposals in parallel
- [X] Keep training NN while evaluating proposals
- [X] Don't add the same proposal multiple times (after counterexamples)
- [ ] Repropose on existing concept 
- [ ] Given a valid inferred regex, parse examples with that regex, choose a matching subgroup, and run inference on that parse

Library:
- [ ] When removing library item, delete other redundant items that are only referenced by it

Trace:
- [X] Pitman-Yor Process
- [ ] Move state (other than observations) into concept itself?
- [ ] Clean up CRPRegexConcept. Currently CRPconcept has no prior on whether it has regex or CRP inside (and concepts not in baseConcepts don't get scored in addConcept)
- [X] Rich concepts get richer (DP?)
- [ ] Move all model logic out of trace.py
- [ ] Make it clearer which functions mutate, and also when a returned trace is clean/dirty
- [ ] concept.py?
- [ ] Refactor CRPConcept (shared functionality between observe and observe_partial)
- [ ] CRPConcept: If there are existing tables with some value, sample rather than choosing the first?
- [ ] ErrorConcept to allow errors?
- [ ] Replace TempDict and TempList with https://pythonhosted.org/pysistence/