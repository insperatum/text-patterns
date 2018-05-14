 sbatch -o ./baselines/128_32_%j.out --qos=tenenbaum --gres=gpu:titan-x:1 --time=180 anaconda-project run python ./make-baseline.py --min_examples=1 --max_examples=1 --hidden_size=128 --embedding_size=32
 sbatch -o ./baselines/64_16_%j.out --qos=tenenbaum --gres=gpu:titan-x:1 --time=180 anaconda-project run python ./make-baseline.py --min_examples=1 --max_examples=1 --hidden_size=64 --embedding_size=16
 sbatch -o ./baselines/32_8_%j.out --qos=tenenbaum --gres=gpu:titan-x:1 --time=180 anaconda-project run python ./make-baseline.py --min_examples=1 --max_examples=1 --hidden_size=32 --embedding_size=8
 sbatch -o ./baselines/16_4_%j.out --qos=tenenbaum --gres=gpu:titan-x:1 --time=180 anaconda-project run python ./make-baseline.py --min_examples=1 --max_examples=1 --hidden_size=16 --embedding_size=4

