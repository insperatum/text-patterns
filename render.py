from collections import Counter
import math

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from graphviz import Digraph

from trace import RegexConcept, PYConcept
import html
import cgi
import numpy as np

import string
alphanumeric = string.ascii_letters + string.digits

def html_escape(s):
	s = cgi.escape(cgi.escape(s))

	#s = s.replace("&amp;lt;", "&lt;").replace("&amp;gt;", "&gt;")
	s = s.replace("\t", "\\t")
	s = s.replace("[", "&#91;").replace("]", "&#93;")

	#s = "".join(x if x in alphanumeric else "&#" + str(ord(x)) + ";" for x in s)
	return s

def saveConcepts(M, filename, onlyIdxs=None, mode="samples"):
	assert(mode in ["samples", "observations", "both"])

	print("Rendering to:%s"%filename)
	trace = M['trace']
	concepts = trace.allConcepts

	dot = Digraph()
	isHidden = {}
	if onlyIdxs is not None:
		for c in concepts:
			isHidden[c] = True
		toAdd = [trace.baseConcepts[i] for i in onlyIdxs]
		while len(toAdd)>0:
			c = toAdd[0]
			toAdd = toAdd[1:]
			isHidden[c] = False
			for c2 in c.conceptsReferenced(trace):
				toAdd.append(c2)

	for concept in concepts:
		samples = [concept.sample(trace) for _ in range(1000)]
		samples_counter = Counter(samples)
		samples.sort(key=samples_counter.get)
		sample_str = ", ".join(list(s if s is not "" else "ε" for s in samples[200:801:200]))
		#tot=sum(samples_counter.values())
		#best = sorted(samples_counter, key=lambda x:math.log(samples_counter.get(x)/tot)/len(x), reverse=True)[:4]
		##unique_samples = set(samples)
		##many_samples = [concept.sample(trace) for _ in range(500)]
		#	
		#if len(samples_counter)>len(best):
		#	sample_str = ", ".join(list(s if s is not "" else "ε" for s in best) + ["..."])
		#else:
		#	sample_str = ", ".join(list(s if s is not "" else "ε" for s in best))
		
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
		isParentRegex = (type(concept) is PYConcept) and (type(trace.getState(concept).baseConcept) is RegexConcept)
		size = 8
		
		total = sum(counter.values())
		nobs=4
		if len(counter)>=nobs:
			sampled_observations = sorted(np.random.choice(list(counter.keys()), p=[x/total for x in counter.values()], replace=False, size=nobs), key=counter.get, reverse=True)
		else:
			sampled_observations = sorted(counter, key=counter.get, reverse=True)
		#sampled_observations = sorted(counter, key=lambda x: math.log(counter[x]/total)/len(x), reverse=True)[:nobs]
		samples = []
		for i in range(100):
			sample = concept.sample(trace)
			if sample not in counter and sample not in samples:
				samples.append(sample)
			if len(sampled_observations) + len(samples)==6:
				break
		if len(counter)>0:
			str_parts = [html_escape(", ".join(list(s if s is not "" else "ε" for s in sampled_observations))) + (", ..." if len(counter)>len(sampled_observations) else "")]
		else:
			str_parts = []
		if len(samples)>0 and isRegex:
			#nRemaining = 5 - len(sampled_observations)
			#nSamples = min(nRemaining, 2)
			nSamples=2
			str_parts.append("<i>(" + html_escape(", ".join(samples[:nSamples])) + (", ..." if len(samples)>nSamples else "") + ")</i>")	
		obs_sample_str = "<br/>".join(str_parts) if len(str_parts)>0 else "(no observations)"

	
		if concept.id==0:
			name_prefix = "<font point-size='%d'><u><b>Alphabet</b></u></font>" % int(size*1.2)
		else:
			name_prefix = "<font point-size='%d'><u><b>"%(int(size*1.5)) + html_escape(concept.str(trace, depth=0)) + "</b></u></font>"

		if isRegex and concept.id != 0:
			content_prefix = "<font point-size='%d'>: "%(int(size*1.5)) + html_escape(concept.str(trace, depth=1, include_self=False)) + "</font>"
		else:
			content_prefix = ""

		nTaskReferences = trace.baseConcept_nTaskReferences.get(concept, 0)
		nConceptReferences = trace.baseConcept_nReferences.get(concept, 0)

		color = "" if isRegex else "lightgrey"
		style = "" if isRegex else "filled"

		if onlyIdxs is None: 
			isHidden[concept] = nTaskReferences<=1 and nConceptReferences==0 and not isRegex and not isParentRegex

		if isHidden[concept]:
			pass
			#dot.node(str(concept.id), "<"
			#+ "<font point-size='%d'>"%int(size*1) + html_escape(obs_str) + "</font>"
			#+ "<font point-size='%d'>"%int(size*1) + obs_sample_str + "</font>"
			#+ ">", color=color, style=style, width='0.2', height='0.2')
		else:				
			dot.node(str(concept.id), "<" 
				+ name_prefix
				+ content_prefix
				+ ("<br/><font point-size='%d'>"%size + html_escape(obs_str) + "</font>" if mode=="observations" else "")
				+ ("<br/><font point-size='%d'>"%size + html_escape(sample_str) + "</font>" if mode=="samples" else "")
				+ ("<br/><font point-size='%d'>"%size + obs_sample_str + "</font>" if mode=="both" else "")
				#+ ("" if nTaskReferences<2 else "<br/><font point-size='%d'>"%size + "(" + ("1 task" if nTaskReferences==1 else "%d tasks" % nTaskReferences) + ")" + "</font>")
				+ ">", color=color, style=style, width='0.5')
		
	for concept in concepts:
		conceptsReferenced = concept.uniqueConceptsReferenced(trace)
		for concept2 in conceptsReferenced:
			if isHidden[concept] or isHidden[concept2]:
				pass
			else:
				color = "black"#"lightgrey" if isHidden[concept] else "black"
				#dot.edge(str(concept2.id), str(concept.id), color=color)
				dot.edge(str(concept.id), str(concept2.id), color=color)

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
	plt.xlabel('Iteration')
	plt.ylabel('NLL')
	plt.savefig(filename)
