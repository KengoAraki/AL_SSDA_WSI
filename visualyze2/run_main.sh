#!/bin/sh

# cv0 (03_G144)
CUDA_VISIBLE_DEVICES=$1 python main.py \
    --title "MME_MF0012_to_MF0003_cl[0, 1, 2]_cv0_init" \
    --output_dir "/home/kengoaraki/Project/DA/SSDA_MME_Saito_WSI/output/result/MME/03_G144/feature_space/" \
    --trg_selected_wsi "03_G144" \
    --G_weight_path "/home/kengoaraki/Project/DA/SSDA_MME_Saito_WSI/output/save_model/source_only/MF0012/cv0/G_cv0_ep9.pth" \
    --F1_weight_path "/home/kengoaraki/Project/DA/SSDA_MME_Saito_WSI/output/save_model/source_only/MF0012/cv0/F_cv0_ep9.pth" \

# cv0 (03_G109-1)
CUDA_VISIBLE_DEVICES=$1 python main.py \
    --title "MME_MF0012_to_MF0003_cl[0, 1, 2]_cv0_init" \
    --output_dir "/home/kengoaraki/Project/DA/SSDA_MME_Saito_WSI/output/result/MME/03_G109-1/feature_space/" \
    --trg_selected_wsi "03_G109-1" \
    --G_weight_path "/home/kengoaraki/Project/DA/SSDA_MME_Saito_WSI/output/save_model/source_only/MF0012/cv0/G_cv0_ep9.pth" \
    --F1_weight_path "/home/kengoaraki/Project/DA/SSDA_MME_Saito_WSI/output/save_model/source_only/MF0012/cv0/F_cv0_ep9.pth" \
