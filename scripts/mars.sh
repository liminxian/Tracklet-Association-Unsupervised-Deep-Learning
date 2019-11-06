#!/usr/bin/env bash
data_dir=/home/liva7/Data
data_set=Mars

CUDA_VISIBLE_DEVICES=1,2,3 python3 taudl_video.py \
                               --data-dir ${data_dir} \
                               -d ${data_set} \
                               -b 384 \
                               -a resnet50 \
                               --features 2048 \
                               --epochs 200 \
                               --num-instances 4 \
                               --start_save 100