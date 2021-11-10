#!/bin/bash
CUDA_VISIBLE_DEVICES=2 nohup python train.py 582220 vqa_struct 64 \
--n_gpus 1 --save_interval 6469 --eval_interval 6469 --n_workers 4 --conf_type vqa_struct \
--model_save_path /files/yxue/research/allstate/out/omninet/vqa_struct_bs64_densenet_features_only_concat_struct_at_the_end \
--move_out_path /files/yxue/research/allstate/out/omninet/vqa_struct_bs64_densenet_features_only_concat_struct_at_the_end < /dev/null > /dev/null 2>&1 &