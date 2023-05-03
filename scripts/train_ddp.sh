#!/bin/bash
PYTHON="python3"
GPU_NUM=$1
export CUDA_VISIBLE_DEVICES=$2
TRAINER=$3
CONFIG=$4
RUN=$5
PORT=$6
PARAMS=${@:7}

export TORCH_DISTRIBUTED_DETAIL=DEBUG

NOW=$(date +"%Y%m%d_%H%M%S")
LOGDIR=$(dirname "$CONFIG")
mkdir -p $LOGDIR/$RUN

LOG_FILE=$LOGDIR/$RUN/${NOW}.txt

$PYTHON -m torch.distributed.launch \
        --nproc_per_node=$GPU_NUM \
        --master_port=$PORT \
        $TRAINER \
        --config $CONFIG --run $RUN \
        $PARAMS

$PYTHON eval.py --config $CONFIG --run $RUN
#        --model_path ${LOGDIR}/${RUN}/checkpoints/best.pth.tar \
#        --save_path $LOGDIR


# ./scripts/train_ddp.sh  4   0,1,2,3   train_cps_mix.py  experiments/cps_mix_v2/_r101/s1323_r101_a100.yaml  s1323_r101_a100  6966