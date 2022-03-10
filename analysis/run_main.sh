#!/bin/sh

# cv0 (03_G144)
CUDA_VISIBLE_DEVICES=$1 python plot_featuremap2.py \
    --title "st1_valt20_srcMF0012_trgMF0003_cl[0, 1, 2]_best_mIoU" \
    --output_dir "/mnt/secssd/AL_SSDA_WSI_MICCAI_strage/st_pretrained_result/featuremap/" \
