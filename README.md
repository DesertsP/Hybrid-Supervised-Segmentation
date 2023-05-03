#  Label-Efficient Hybrid-Supervised Learning for Medical Image Segmentation
This is the official implementaion of paper " Label-Efficient Hybrid-Supervised Learning for Medical Image Segmentation".

**This code is messy and unsorted now, we will clean it up in the future.**

This repository contains PyTorch training and evaluation code for Pascal VOC 2012 and COCO dataset.



```bash
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=2 train_xx.py --config configs/xx.yaml
``` 