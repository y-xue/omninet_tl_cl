#!/bin/bash


# CUDA_VISIBLE_DEVICES=2 python saliency.py 420549 vqa_struct 64 --n_gpus 1 \
# --save_interval 100000 --eval_interval 100000 --conf_type vqa_struct \
# --model_save_path /files/yxue/research/allstate/out/omninet/vqa_struct_norm_bs64_clustering_std3_inject_struct_at_logits_608085 \
# --move_out_path /files/yxue/research/allstate/out/omninet/vqa_struct_norm_bs64_clustering_std3_inject_struct_at_logits_608085 \
# --structured_folder synthetic_structured_clustering_std3 --inject_at_logits \
# --no_logit_struct_peripheral \
# --n_workers 2 --init_lr 0.01 --restore_last --split tr; # < /dev/null > /dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=2 python saliency.py 420549 vqa_struct 64 --n_gpus 1 \
# --save_interval 100000 --eval_interval 100000 --conf_type vqa_struct \
# --model_save_path /files/yxue/research/allstate/out/omninet/vqa_struct_norm_bs64_clustering_std3_inject_struct_at_logits_608085 \
# --move_out_path /files/yxue/research/allstate/out/omninet/vqa_struct_norm_bs64_clustering_std3_inject_struct_at_logits_608085 \
# --structured_folder synthetic_structured_clustering_std3 --inject_at_logits \
# --no_logit_struct_peripheral \
# --n_workers 2 --init_lr 0.01 --restore_last --split val;


CUDA_VISIBLE_DEVICES=2 python saliency.py 420549 vqa_struct 64 --n_gpus 1 \
--save_interval 100000 --eval_interval 100000 --conf_type vqa_struct \
--model_save_path /files/yxue/research/allstate/out/omninet/struct_domain/vqa_struct_norm_bs64_dp0.8_init_lr0.01_414079 \
--move_out_path /files/yxue/research/allstate/out/omninet/struct_domain/vqa_struct_norm_bs64_dp0.8_init_lr0.01_414079 \
--structured_folder synthetic_structured_clustering_std3 --inject_at_logits --inject_at_encoder \
--no_logit_struct_peripheral \
--n_workers 2 --init_lr 0.01 --struct_periph_dropout 0.1 --struct_temp_periph_dropout 0.8 \
--struct_spat_periph_dropout 0.8 --restore_last --split tr; # < /dev/null > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=2 python saliency.py 420549 vqa_struct 64 --n_gpus 1 \
--save_interval 100000 --eval_interval 100000 --conf_type vqa_struct \
--model_save_path /files/yxue/research/allstate/out/omninet/struct_domain/vqa_struct_norm_bs64_dp0.8_init_lr0.01_414079 \
--move_out_path /files/yxue/research/allstate/out/omninet/struct_domain/vqa_struct_norm_bs64_dp0.8_init_lr0.01_414079 \
--structured_folder synthetic_structured_clustering_std3 --inject_at_logits --inject_at_encoder \
--no_logit_struct_peripheral \
--n_workers 2 --init_lr 0.01 --struct_periph_dropout 0.1 --struct_temp_periph_dropout 0.8 \
--struct_spat_periph_dropout 0.8 --restore_last --split val;