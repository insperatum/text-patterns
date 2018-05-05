from collections import Counter

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from graphviz import Digraph

from trace import RegexConcept
import html
import numpy as np

def saveConcepts(M, filename):
	print("Rendering to:%s"%filename)
	trace = M['trace']
	concepts = trace.allConcepts

	dot = Digraph()
	isMini = {}
	for concept in concepts:
#		samples = [concept.sample(trace) for _ in range(5)]
#		unique_samples = set(samples)
#		many_samples = [concept.sample(trace) for _ in range(500)]
			
#		if any(x not in unique_samples for x in many_samples):
#			sample_str = ", ".join(list(s if s is not "" else "" for s in unique_samples) + ["..."])
#		else:
#			sample_str = ", ".join(list(s if s is not "" else "" for s in unique_samples))
		
		observations = concept.get_observations(trace)
		counter = Counter(observations)	
		if len(counter)>5:
			total = sum(counter.values())
			sampled_observations = np.random.choice(counter.keys(), p=[x/total for x in counter.values()], replace=False)
			obs_str = ", ".join(list(s if s is not "" else "" for s in sampled_observations) + ["..."])
		else:
			obs_str = ", ".join(list(s if s is not "" else "" for s in counter))
		
		
		isRegex = type(concept) is RegexConcept
		size = 8
		
		name_prefix = "<font point-size='%d'>"%(int(size*1.5)) + html.escape(concept.str(trace, depth=1)) + "</font><br/>"
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
				#+ "<font point-size='%d'>"%size + html.escape(sample_str) + "</font>"
				+ "<font point-size='%d'>"%size + html.escape(obs_str) + "</font>"
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

	interval=20
	xs = range(interval, M['state']['iteration']+1, interval)
	ys = [np.mean(M['state']['network_losses'][i-interval:i]) for i in xs]

	plt.xlim(xmin=0, xmax=M['state']['iteration']+1)

	for iteration in M['state']['task_iterations']:
		plt.axvline(x=iteration, color='r', linewidth=0.5)

	plt.plot(xs, ys)

	# plt.ylim(ymin=0, ymax=25)
	plt.xlabel('iteration')
	plt.ylabel('NLL')
	plt.savefig(filename)
