import loader
import os

for model in ('results/%s'%x for x in os.listdir('results') if x[-3:]==".pt"):
	print("Loading", model)
	M = loader.load(model)
	loader.saveRender(M)
