#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
NGPUS=$1
GPUS=$2
RUN=$3
CONFIG=$4
PORT=$5

export CUDA_VISIBLE_DEVICES=$GPUS
mkdir -p logs/$RUN

# use torch.distributed.launch
python -m torch.distributed.launch \
    --nproc_per_node=$NGPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=$PORT \
    train_sup.py --config $CONFIG --seed 2 --port $PORT 2>&1 | tee logs/${RUN}/${now}.txt
