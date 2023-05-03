pip install pkgs/chainer-7.7.0 pkgs/pydensecrf-master \
   pkgs/Pillow-8.0.1-cp36-cp36m-manylinux1_x86_64.whl pkgs/chainercv-0.13.1

python tools/infer_voc.py --config experiments/gather_ocr_grad_fix_tau1_reg10.yaml \
 --output outputs/gather_ocr_grad_fix_tau1_reg10/12081206/train_iter1_gt1_fp02_bg3 \
 --checkpoint outputs/gather_ocr_grad_fix_tau1_reg10/12081206/checkpoint/checkpoint20.pth.tar \
 --data_root /private/DATASETS/VOC+SBD/VOCdevkit/VOC2012 \
 --data_list voc12/train.txt \
 --bg_pow 3 \
 --use_cls_label 1 \
 --cluster_iters 1 \
 --co_cluster 0
python tools/eval.py --output outputs/gather_ocr_grad_fix_tau1_reg10/12081206/train_iter1_gt1_fp02_bg3/pred --data_root '/private/DATASETS/VOC+SBD/VOCdevkit/VOC2012' --split train
python tools/eval.py --output outputs/gather_ocr_grad_fix_tau1_reg10/12081206/train_iter1_gt1_fp02_bg3/crf --data_root '/private/DATASETS/VOC+SBD/VOCdevkit/VOC2012' --split train