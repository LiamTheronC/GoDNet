#!/usr/bin/env bash

TYPE_FEATS=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py --type-feats $TYPE_FEATS --num-gpus $GPUS
