#!/bin/bash

cuda_device=1;
move_out=/files/yxue/research/allstate/out;
# server_name=wcdl1;
save_intvl=500;
eval_intvl=500;
bs=16;
n_iter=6005;
seq=ITTIV;
task=mm_ITV;
n_gpus=1;
all_seed=(1029 47 816 21 219 222 628 845 17 531 635);
#random_seed=(983 422 84 28 48 714 935 664 844 261);
#numpy_seed=(1024 491 290 809 15 403 322 1 939 888);

i=0;
sw=0;
CUDA_VISIBLE_DEVICES=${cuda_device} python train.py ${n_iter} ${task} ${bs} --n_gpus ${n_gpus} \
--save_interval ${save_intvl} --eval_interval ${eval_intvl} \
--model_save_path /out/mm_nodup_rewarding_random_seq/${seq}_sample_weights${sw}_${bs}bs_${n_iter}it_$((all_seed[$i])) \
--move_out_path ${move_out}/mm_nodup_rewarding_random_seq/${seq}_sample_weights${sw}_${bs}bs_${n_iter}it_$((all_seed[$i])) \
--sample_idx_fn sample_idx_ITTIV_pT.7_no_dup_img_text_video_seed10 \
--sample_weights_fn sample_weights_${seq}_${sw}.json --peripherals_type timeline --conf_type timeline \
--all_seed $((all_seed[$i])) --rewarding --random_seq > /mm_nodup_rewarding_random_seq_${seq}_sample_weights${sw}_${bs}bs_${n_iter}it_$((all_seed[$i])).log;
rm -r /out/mm_nodup_rewarding_random_seq/${seq}_sample_weights${sw}_${bs}bs_${n_iter}it_$((all_seed[$i]));
sleep 10;

