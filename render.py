import matplotlib
matplotlib.use("Agg")
# import networkx as nx
import matplotlib.pyplot as plt
from graphviz import Digraph

import trace
import loader
from collections import Counter
import html
# print("Loading model")
# M = loader.load("./models/train.py_dot2_1523756200938187.pt")

def saveConcepts(M, filename):
	trace = M['trace']
	concepts = trace.baseConcepts
	print("Making graph")

	dot = Digraph()
	for concept in concepts:
		name = concept.str(trace, short=True)
		c = Counter(concept.sample(trace) for _ in range(1000))
		samples = sorted(c, key=c.get)
		if len(samples)<=4:
			sample_str = ", ".join(samples[:4])
		else:
			sample_str = ", ".join(samples[:3] + ["..."])
		dot.node(str(concept.id), "<" + html.escape(name) + "<br/>" + "<FONT POINT-SIZE='8'>" + html.escape(sample_str) + "</FONT>" + ">")
	for concept in concepts:
		conceptsReferenced = concept.uniqueConceptsReferenced(trace)
		for concept2 in conceptsReferenced:
			dot.edge(str(concept2.id), str(concept.id))

	dot.format = 'pdf'
	dot.render(filename)  

def saveTrainingError(M, filename):
	plt.clf()
	plt.plot(range(1, M['state']['iteration']+1), M['state']['network_losses'])
	plt.xlim(xmin=0, xmax=M['state']['iteration']+1)
	# plt.ylim(ymin=0, ymax=25)
	plt.xlabel('iteration')
	plt.ylabel('NLL')
	plt.savefig(filename)