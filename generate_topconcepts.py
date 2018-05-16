import loader


nSamples=3

model = "results/model.pt"

print("\nModel:", model)
M = loader.load(model)
trace = M['trace']

def latexify(s):
	return "\\verb|" + s + "|"

for c in sorted(trace.baseConcepts, key=lambda c: trace.baseConcept_nReferences[c], reverse=True):
	samples = [c.sample(trace) for _ in range(5)]
	print(c + "& " +  latexify(", ".join(samples)) + "\\")

#print()
#for model in models:
#	print(results[model])
	
