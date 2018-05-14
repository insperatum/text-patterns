import loader
#from collections import Counter
#import model
#import pregex as pre

data, group_idxs, test_data = loader.loadData("./data/csv.p", n_examples=1000, n_tasks=50, max_length=15)
#M = loader.load('./models/task38.pt')
#net = M['net']
#trace = M['trace']
#concepts = trace.baseConcepts

#r = pre.create("(NA)|(NA)")
#print(trace.model.scoreregex(r, trace))
# for concept in concepts:
# 	print(str(concept))
# 	# c = Counter(concept.sample(trace) for _ in range(1000))
# 	# samples = sorted(c, key=c.get, reverse=True)
# 	# print(samples)
# 	# print()

for i in range(len(test_data)): print(i, list(set(test_data[i]))[:5])

print(len(data), "train +", len(test_data), "test =", len(data) + len(test_data), "total")
