#!/bin/bash
steps=(6469 12938 19407 25876 32345 38814 45283);

for i in 0 1 2 3 4 5 6
do
	CUDA_VISIBLE_DEVICES=1 python train.py 582220 vqa_struct 64 --n_gpus 1 \
	--save_interval 6469 --eval_interval 6469 --n_workers 4 --conf_type vqa_struct \
	--model_save_path /files/yxue/research/allstate/out/omninet/vqa_struct_bs64_densenet_features \
	--move_out_path /files/yxue/research/allstate/out/omninet/vqa_struct_bs64_densenet_features \
	--restore $((steps[$i])) --eval_first --evaluate > /evaluating_$((steps[$i])).log;
	sleep 10;
done

# CUDA_VISIBLE_DEVICES=2 nohup python train.py 582220 vqa_struct 64 --n_gpus 1 \
# --save_interval 6469 --eval_interval 6469 --n_workers 4 --conf_type vqa_struct \
# --model_save_path /files/yxue/research/allstate/out/omninet/vqa_struct_bs64_densenet_features \
# --move_out_path /files/yxue/research/allstate/out/omninet/vqa_struct_bs64_densenet_features \
# --restore_last < /dev/null > /dev/null 2>&1 &

