#!/bin/bash

export PYTHONPATH=`pwd`

# python bin/download_tabrepo_datasets.py

for i in 1 3 5 7 9
do
    echo "Running iteration $i"
    export SEED=$i
    # python bin/train_synthetic.py --config-path=config/synthetic/dataset2vec.yaml
    # python bin/train_synthetic.py --config-path=config/synthetic/gbdsim.yaml
    python bin/train_uci.py --config-path=config/uci/dataset2vec.yaml
    python bin/train_uci.py --config-path=config/uci/gbdsim.yaml
    # python bin/train_tabrepo.py --config-path=config/tabrepo/gbdsim.yaml
    # python bin/train_tabrepo.py --config-path=config/tabrepo/dataset2vec.yaml
    clear
done
