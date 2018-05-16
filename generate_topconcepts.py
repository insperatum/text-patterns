import loader


nSamples=10

model = "results/model.pt"

print("\nModel:", model)
M = loader.load(model)
trace = M['trace']

def latexify(s):
	return "\\verb|" + s + "|"

print("\\begin{tabular}{r|" + " ".join("l"*nSamples) + "}")
print("\\textbf{\\# Times referenced}&\\\\")
print("\\hline")
for c in sorted(trace.baseConcepts, key=lambda c: trace.baseConcept_nReferences.get(c,0), reverse=True)[:10]:
	samples = [c.sample(trace) for _ in range(nSamples)]
	print(trace.baseConcept_nReferences.get(c,0), "&", latexify(", ".join(samples)) + "\\\\")
print("\\end{tabular}")
#print()
#for model in models:
#	print(results[model])
	
