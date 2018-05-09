from collections import Counter

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from graphviz import Digraph

from trace import RegexConcept
import html
import numpy as np

def html_escape(s):
	s = html.escape(s)
	s = s.replace("{", "&{;")
	return s

def saveConcepts(M, filename):
	print("Rendering to:%s"%filename)
	trace = M['trace']
	concepts = trace.allConcepts

	dot = Digraph()
	isMini = {}
	for concept in concepts:
		samples = [concept.sample(trace) for _ in range(5)]
		unique_samples = set(samples)
		many_samples = [concept.sample(trace) for _ in range(500)]
			
		if any(x not in unique_samples for x in many_samples):
			sample_str = ", ".join(list(s if s is not "" else "ε" for s in unique_samples) + ["..."])
		else:
			sample_str = ", ".join(list(s if s is not "" else "ε" for s in unique_samples))
		
		observations = concept.get_observations(trace)
		counter = Counter(observations)	
		if len(counter)>5:
			total = sum(counter.values())
			sampled_observations = np.random.choice(list(counter.keys()), p=[x/total for x in counter.values()], replace=False, size=4)
			obs_str = ", ".join(list(s if s is not "" else "ε" for s in sampled_observations) + ["..."])
		elif len(counter)>0:
			obs_str = ", ".join(list(s if s is not "" else "ε" for s in counter))
		else:
			obs_str = "(no observations)"
		
		
		isRegex = type(concept) is RegexConcept
		size = 8
	
		if concept.id==0:
			name_prefix = "<font point-size='%d'><u><b>Alphabet</b></u></font>" % int(size*1.2)
		else:
			name_prefix = "<font point-size='%d'><u><b>"%(int(size*1.5)) + html_escape(concept.str(trace, depth=0)) + "</b></u></font>"

		if isRegex and concept.id != 0:
			content_prefix = "<br/><font point-size='%d'>"%(int(size*1.5)) + html_escape(concept.str(trace, depth=1, include_self=False)) + "</font>"
		else:
			content_prefix = ""

		nTaskReferences = trace.baseConcept_nTaskReferences.get(concept, 0)
		nConceptReferences = trace.baseConcept_nReferences.get(concept, 0)

		color = "" if isRegex else "lightgrey"
		style = "" if isRegex else "filled"

		isMini[concept] = nTaskReferences<=1 and nConceptReferences==0 and not isRegex

		if isMini[concept]:
			dot.node(str(concept.id), "<"
				+ "<font point-size='%d'>"%int(size*0.7) + html_escape(obs_str) + "</font>"
				+ ">", color=color, style=style, width='0.2', height='0.2')
		else:				
			dot.node(str(concept.id), "<" 
				+ name_prefix
				+ content_prefix
				+ "<br/><font point-size='%d'>"%size + html_escape(obs_str) + "</font>"
				+ "<br/><font point-size='%d'><i>"%size + html.escape(sample_str) + "</i></font>"
				#+ ("" if nTaskReferences<2 else "<br/><font point-size='%d'>"%size + "(" + ("1 task" if nTaskReferences==1 else "%d tasks" % nTaskReferences) + ")" + "</font>")
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
