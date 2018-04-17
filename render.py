import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from graphviz import Digraph

import trace
import loader
from collections import Counter
import html
import numpy as np

def saveConcepts(M, filename):
	trace = M['trace']
	concepts = trace.baseConcepts

	dot = Digraph()
	for concept in concepts:
		name = concept.str(trace, short=True)
		c = Counter(concept.sample(trace) for _ in range(1000))
		# c = Counter(x.value for x in trace.getState(concept).observations)
		samples = sorted(c, key=c.get, reverse=True)
		if len(samples)<=4:
			sample_str = ", ".join(samples[:4])
		else:
			sample_str = ", ".join(samples[:3] + ["..."])
		
		if not any(concept in parent.uniqueConceptsReferenced(trace) for parent in concepts):
			dot.node(str(concept.id), "<" + html.escape(name) + "<br/>" + "<FONT POINT-SIZE='8'>" + html.escape(sample_str) + "</FONT>" + ">", color="lightgrey")
		else:
			dot.node(str(concept.id), "<" + html.escape(name) + "<br/>" + "<FONT POINT-SIZE='8'>" + html.escape(sample_str) + "</FONT>" + ">")

	for concept in concepts:
		conceptsReferenced = concept.uniqueConceptsReferenced(trace)
		for concept2 in conceptsReferenced:
			dot.edge(str(concept2.id), str(concept.id))

	dot.format = 'pdf'
	dot.render(filename)  

def saveTrainingError(M, filename):
	plt.clf()

	interval=20
	xs = range(1, M['state']['iteration']+1, interval)
	ys = [np.mean(M['state']['network_losses'][i:i+50]) for i in range(0, len(M['state']['network_losses']), interval)]
	plt.plot(xs, ys)

	plt.xlim(xmin=0, xmax=M['state']['iteration']+1)
	# plt.ylim(ymin=0, ymax=25)
	plt.xlabel('iteration')
	plt.ylabel('NLL')
	plt.savefig(filename)