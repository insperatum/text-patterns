hidden=256
embedding=64
for shot in 1 5; do
	sbatch -o ./baselines/h${hidden}_e${embedding}_${shot}shot%j.out --qos=tenenbaum --gres=gpu:titan-x:1 --time=180 anaconda-project run python ./train-pp.py --min_examples=$shot --max_examples=$shot --hidden_size=$hidden --embedding_size=embedding
	for f in results/*00.pt; do
		sbatch -o ./baselines/h${hidden}_e${embedding}_${shot}shot%j.out --qos=tenenbaum --gres=gpu:titan-x:1 --time=180 anaconda-project run python ./train-pp.py --min_examples=$shot --max_examples=$shot --hidden_size=$hidden --embedding_size=embedding --mode=model --model_file=$f
	done
done
