import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from graphviz import Digraph

from trace import RegexConcept
import loader
from collections import Counter
import html
import numpy as np

def saveConcepts(M, filename):
	print("Rendering to:%s"%filename)
	trace = M['trace']
	concepts = trace.baseConcepts

	dot = Digraph()
	isMini = {}
	for concept in concepts:
		n=1000
		c = Counter(concept.sample(trace) for _ in range(n))
		# c = Counter(x.value for x in trace.getState(concept).observations)
		samples = sorted(c, key=c.get, reverse=True)
		samples = [x for x in samples if c.get(x) >= c.get(samples[0])/50]
		if len(samples)<=4:
			sample_str = ", ".join(samples[:5])
		else:
			sample_str = ", ".join(samples[:4] + ["..."])
		
		
		isRegex = type(concept) is RegexConcept
		size = 8
		
		# name_prefix = "<font point-size='%d'>"%(int(size*1.5)) + html.escape(concept.str(trace, short=True)) + "</font><br/>" if isRegex else ""
		name_prefix = "<font point-size='%d'>"%(int(size*1.5)) + html.escape(concept.str(trace, short=True)) + "</font><br/>"
		nTaskReferences = trace.baseConcept_nTaskReferences.get(concept, 0)
		nConceptReferences = trace.baseConcept_nReferences.get(concept, 0)

		color = "" if isRegex else "lightgrey"
		style = "" if isRegex else "filled"

		isMini[concept] = nTaskReferences<=1 and nConceptReferences==0 and not isRegex

		if isMini[concept]:
			dot.node(str(concept.id), "", color=color, style=style, width='0.2', height='0.2')
		else:				
			dot.node(str(concept.id), "<" 
				+ name_prefix
				+ "<font point-size='%d'>"%size + html.escape(sample_str) + "</font>"
				+ ("" if nTaskReferences<2 else "<br/><font point-size='%d'>"%size + "(" + ("1 task" if nTaskReferences==1 else "%d tasks" % nTaskReferences) + ")" + "</font>")
				+ ">", color=color, style=style, width='0.5')
		
	for concept in concepts:
		conceptsReferenced = concept.uniqueConceptsReferenced(trace)
		for concept2 in conceptsReferenced:
			color = "lightgrey" if isMini[concept] else "black"
			dot.edge(str(concept2.id), str(concept.id), color=color)

	dot.format = 'pdf'
	dot.render(filename)  

def saveTrainingError(M, filename):
	plt.clf()

	interval=25
	xs = range(1, M['state']['iteration']+1, interval)
	ys = [np.mean(M['state']['network_losses'][i:i+interval]) for i in range(0, len(M['state']['network_losses']), interval)]
	plt.plot(xs, ys)

	plt.xlim(xmin=0, xmax=M['state']['iteration']+1)

	for iteration in M['state']['task_iterations']:
		plt.axvline(x=iteration, color='r')
	# plt.ylim(ymin=0, ymax=25)
	plt.xlabel('iteration')
	plt.ylabel('NLL')
	plt.savefig(filename)