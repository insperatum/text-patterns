import loader

import render

nSamples=6

model = "results/model.pt"

print("\nModel:", model)
M = loader.load(model)
trace = M['trace']

def latexify(s):
	return "\\verb|" + s + "|"

topConcepts = sorted(trace.baseConcepts, key=lambda c: trace.baseConcept_nReferences.get(c,0), reverse=True)[:10]
print("\\begin{tabular}{l|" + " ".join("l"*nSamples) + "}")
#print("&\\textbf{Reuses} & \\textbf{Samples}\\\\")
print(" & Samples\\\\")
print("\\hline")
for c in topConcepts: 
	samples = [c.sample(trace) for _ in range(nSamples)]
	#print(trace.baseConcept_nReferences.get(c,0), "&", latexify(", ".join(samples)) + "\\\\")
	print("\\textbf{\\#" + str(c.id) + "}", "&", latexify(", ".join(samples)) + "\\\\")
print("\\end{tabular}")

bestChildren = [max((x for x in trace.baseConcepts if c in x.conceptsReferenced(trace)), key=lambda x: trace.baseConcept_nReferences.get(c,)) for c in topConcepts]
render.saveConcepts(M, M['save_to']+"render_topconcepts.gv", onlyIdxs=[x.id for x in set(topConcepts + bestChildren)], mode="observations")
#print()
#for model in models:
#	print(results[model])
	
