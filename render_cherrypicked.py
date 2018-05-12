import loader
import render
import argparse

#for model in ('results/%s'%x for x in os.listdir('results') if x[-3:]==".pt"):
#	print("Loading", model)
#	M = loader.load(model)
#loader.saveRender(M)
parser = argparse.ArgumentParser()
parser.add_argument('--i', type=str, default="2,21,24,26,34,35,45,47")
args = parser.parse_args()

M = loader.load("results/model.pt")
onlyIdxs = [int(x) for x in args.i.split(",")]
render.saveConcepts(M, M['save_to'] + "render_cherrypicked.gv", onlyIdxs=onlyIdxs)
print("finished.")
