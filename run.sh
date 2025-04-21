#!/bin/bash

export PYTHONPATH=`pwd`

for i in 1 3 5 6 7
do
    echo "Running iteration $i"
    export SEED=$i
    python bin/train_synthetic.py --config-path=config/synthetic/dataset2vec.yaml
    python bin/train_synthetic.py --config-path=config/synthetic/gbdsim.yaml
    python bin/train_uci.py --config-path=config/uci/dataset2vec.yaml
    python bin/train_uci.py --config-path=config/uci/gbdsim.yaml
    clear
done