#!/bin/bash
cuda_device=3;
move_out=/files/yxue/research/allstate/out;

save_intvl=500;
eval_intvl=500;
bs=16;
n_iter=10005;
seq=ITTIV;
task=mm_ITV;
n_gpus=1;
all_seed=(1029 47 816 21 219 222 628 845 17 531 635);
dropout1=0.5;
dropout2=0.25;

sw=0;
ot=0.0001;
os=1000;

for i in 0
do
	CUDA_VISIBLE_DEVICES=${cuda_device} python train.py ${n_iter} ${task} ${bs} --n_gpus ${n_gpus} \
	--save_interval ${save_intvl} --eval_interval ${eval_intvl} --save_best \
	--model_save_path /out/mm/ITTIV_all/only1-2-0/mm_only1-2-0_random_seq_${seq}_sample_weights${sw}_${bs}bs_${n_iter}it_${eval_intvl}eval_intvl_$((all_seed[$i]))_dropout_${dropout1}_${dropout2} \
	--move_out_path ${move_out}/mm/ITTIV_all/only1-2-0/mm_only1-2-0_random_seq_${seq}_sample_weights${sw}_${bs}bs_${n_iter}it_${eval_intvl}eval_intvl_$((all_seed[$i]))_dropout_${dropout1}_${dropout2} \
	--sample_idx_fn sample_idx_ITTIV_pT.7_no_dup_all_img_text_video_seed10 \
	--sample_weights_fn sample_weights_${seq}_${sw}.json --peripherals_type timeline --conf_type timeline \
	--all_seed $((all_seed[$i])) --rewarding --pick_seq random \
	--skip_seqs 1-0-0 1-1-0 2-2-0 2-0-0 2-1-0 2-2-1 0-1-0 1-1-1 0-2-0 0-0-1 > /mm_only1-2-0_random_seq_${seq}_sample_weights${sw}_${bs}bs_${n_iter}it_${eval_intvl}eval_intvl_$((all_seed[$i]))_dropout_${dropout1}_${dropout2}.log;
	rm -r /out/mm/ITTIV_all/only1-2-0/mm_random_seq_${seq}_sample_weights${sw}_${bs}bs_${n_iter}it_${eval_intvl}eval_intvl_$((all_seed[$i]))_dropout_${dropout1}_${dropout2};
	sleep 10;
done

