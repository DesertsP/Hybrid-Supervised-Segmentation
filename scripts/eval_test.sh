#!/bin/bash

EXP=$1;
RUN=$2;

CONFIG=experiments/voc_semi/$EXP.yaml
CHECKPOINTS=outputs/$EXP/$RUN/checkpoint;

START=$3;
END=$4;

BG_POW=$5;
FP_CUT=$6;
USE_GT=$7;
NUM_ITERS=$8;
CO_CLUSTER=$9;

OUTPUT_DIR=outputs/$EXP/$RUN/test_results_gt${USE_GT}_fp${FP_CUT}_bg${BG_POW}_siam_iters${NUM_ITERS};

DATAROOT='/root/datasets/VOCdevkit/VOC2012';


for((i=$START;i<=$END;i++));
do
echo '================================'
CKP=$CHECKPOINTS/checkpoint$i.pth.tar;
echo $CKP;

python infer_siam.py --config $CONFIG \
       --output $OUTPUT_DIR/$i \
       --checkpoint $CKP \
       --bg_pow $BG_POW \
       --fp_cut $FP_CUT \
       --use_cls_label $USE_GT \
       --data_root $DATAROOT \
       --cluster_iters $NUM_ITERS \
       --data_list voc12/test.txt \
       --co_cluster $CO_CLUSTER;
done
