#!/bin/bash
cuda_device=3;
move_out=/files/yxue/research/allstate/out;

save_intvl=100;
eval_intvl=100;
bs=16;
n_iter=2005;
seq=ITTIV;
task=mm_ITV;
n_gpus=1;
all_seed=(1029 47 816 21 219 222 628 845 17 531 635);
dropout1=0.5;
dropout2=0.25;

sw=0;

for i in 1 2 3 4
do
	CUDA_VISIBLE_DEVICES=${cuda_device} python train.py ${n_iter} ${task} ${bs} --n_gpus ${n_gpus} \
	--save_interval ${save_intvl} --eval_interval ${eval_intvl} --save_best \
	--model_save_path /out/mm/random_seq/mm_random_seq_${seq}_sample_weights${sw}_${bs}bs_${n_iter}it_$((all_seed[$i]))_dropout_${dropout1}_${dropout2} \
	--move_out_path ${move_out}/mm/random_seq/mm_random_seq_${seq}_sample_weights${sw}_${bs}bs_${n_iter}it_$((all_seed[$i]))_dropout_${dropout1}_${dropout2} \
	--sample_idx_fn sample_idx_ITTIV_pT.7_no_dup_img_text_video_seed10 \
	--sample_weights_fn sample_weights_${seq}_${sw}.json --peripherals_type timeline --conf_type timeline \
	--all_seed $((all_seed[$i])) --rewarding --pick_seq random > /mm_random_seq_${seq}_sample_weights${sw}_${bs}bs_${n_iter}it_$((all_seed[$i]))_dropout_${dropout1}_${dropout2}.log;
	rm -r /out/mm/random_seq/mm_random_seq_${seq}_sample_weights${sw}_${bs}bs_${n_iter}it_$((all_seed[$i]))_dropout_${dropout1}_${dropout2};
	sleep 10;
done
