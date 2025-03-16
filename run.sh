#!/bin/bash

export PYTHONPATH=`pwd`

# python bin/train_synthetic.py --config-path=config/synthetic/dataset2vec.yaml
python bin/train_synthetic.py --config-path=config/synthetic/gbdsim.yaml
# python bin/train_uci.py --config-path=config/uci/dataset2vec.yaml
# python bin/train_uci.py --config-path=config/uci/gbdsim.yaml