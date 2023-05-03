#!/bin/bash
PYTHON="python3"
CONFIG=$1
RUN=$2

LOGDIR=$(dirname "$CONFIG")
mkdir -p $LOGDIR/$RUN


$PYTHON eval_seg.py --config $CONFIG \
        --model_path ${LOGDIR}/${RUN}/checkpoints/best.pth.tar \
        --save_path $LOGDIR


# ./scripts/train_ddp.sh  4   0,1,2,3   train_cps_mix.py  experiments/cps_mix_v2/_r101/s1323_r101_a100.yaml  s1323_r101_a100  6966