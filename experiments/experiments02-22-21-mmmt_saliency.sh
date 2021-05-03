#!/bin/bash

# CUDA_VISIBLE_DEVICES=7 nohup python saliency.py 420549 vqa_struct 64 --n_gpus 1 \
# --save_interval 100000 --eval_interval 100000 --conf_type vqa_struct \
# --model_save_path /files/yxue/research/allstate/out/omninet/struct_domain/vqa_struct_norm_bs64_dp0.8_init_lr0.01_saliency \
# --move_out_path /files/yxue/research/allstate/out/omninet/struct_domain/vqa_struct_norm_bs64_dp0.8_init_lr0.01_saliency \
# --structured_folder synthetic_structured_clustering_std3 --inject_at_logits --inject_at_encoder \
# --n_workers 2 --init_lr 0.01 --struct_periph_dropout 0.1 --struct_temp_periph_dropout 0.8 \
# --struct_spat_periph_dropout 0.8 --restore_last < /dev/null > /dev/null 2>&1 &


# CUDA_VISIBLE_DEVICES=0 nohup python saliency.py 420549 vqa_struct 64 --n_gpus 1 \
# --save_interval 100000 --eval_interval 100000 --conf_type vqa_struct \
# --model_save_path /files/yxue/research/allstate/out/omninet/struct_domain/vqa_struct_norm_bs64_dp0.8_init_lr0.01 \
# --move_out_path /files/yxue/research/allstate/out/omninet/struct_domain/vqa_struct_norm_bs64_dp0.8_init_lr0.01 \
# --structured_folder synthetic_structured_clustering_std3 --inject_at_logits --inject_at_encoder \
# --n_workers 2 --init_lr 0.01 --struct_periph_dropout 0.1 --struct_temp_periph_dropout 0.8 \
# --struct_spat_periph_dropout 0.8 --restore 414079 --split val < /dev/null > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=1 python saliency.py 420549 vqa_struct 64 --n_gpus 1 \
--save_interval 100000 --eval_interval 100000 --conf_type vqa_struct \
--model_save_path /files/yxue/research/allstate/out/omninet/struct_domain/vqa_struct_norm_bs64_dp0.8_init_lr0.01 \
--move_out_path /files/yxue/research/allstate/out/omninet/struct_domain/vqa_struct_norm_bs64_dp0.8_init_lr0.01 \
--structured_folder synthetic_structured_clustering_std3 --inject_at_logits --inject_at_encoder \
--n_workers 2 --init_lr 0.01 --struct_periph_dropout 0.1 --struct_temp_periph_dropout 0.8 \
--struct_spat_periph_dropout 0.8 --restore 414079 --split tr; # < /dev/null > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=1 python saliency.py 420549 vqa_struct 64 --n_gpus 1 \
--save_interval 100000 --eval_interval 100000 --conf_type vqa_struct \
--model_save_path /files/yxue/research/allstate/out/omninet/struct_domain/vqa_struct_norm_bs64_dp0.8_init_lr0.01 \
--move_out_path /files/yxue/research/allstate/out/omninet/struct_domain/vqa_struct_norm_bs64_dp0.8_init_lr0.01 \
--structured_folder synthetic_structured_clustering_std3 --inject_at_logits --inject_at_encoder \
--n_workers 2 --init_lr 0.01 --struct_periph_dropout 0.1 --struct_temp_periph_dropout 0.8 \
--struct_spat_periph_dropout 0.8 --restore 414079 --split val;

# CUDA_VISIBLE_DEVICES=0 nohup python saliency.py 840980 vqa_struct 64 --n_gpus 1 \
# --save_interval 100000 --eval_interval 100000 --conf_type vqa_struct \
# --model_save_path /files/yxue/research/allstate/out/omninet/vqa_struct_norm_bs64_encoder_fusion_gating_per_frame_softmax_convex_679349 \
# --move_out_path /files/yxue/research/allstate/out/omninet/vqa_struct_norm_bs64_encoder_fusion_gating_per_frame_softmax_convex_679349 \
# --structured_folder synthetic_structured_clustering_std3 --inject_at_logits --inject_at_encoder \
# --temp_fusion_attn_type selective --spat_fusion_attn_type selective --convex_gate --n_workers 2 \
# --fusion --restore_last --split val < /dev/null > /dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python saliency.py 840980 vqa_struct 64 --n_gpus 1 \
# --save_interval 100000 --eval_interval 100000 --conf_type vqa_struct \
# --model_save_path /files/yxue/research/allstate/out/omninet/vqa_struct_norm_bs64_encoder_fusion_gating_per_frame_softmax_convex_679349 \
# --move_out_path /files/yxue/research/allstate/out/omninet/vqa_struct_norm_bs64_encoder_fusion_gating_per_frame_softmax_convex_679349 \
# --structured_folder synthetic_structured_clustering_std3 --inject_at_logits --inject_at_encoder \
# --temp_fusion_attn_type selective --spat_fusion_attn_type selective --convex_gate --n_workers 2 \
# --fusion --restore_last --split tr < /dev/null > /dev/null 2>&1 &