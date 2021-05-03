#!/bin/bash
out_path=/files/yxue/research/allstate/out;

save_intvl=500;
eval_intvl=500;
bs=16;
n_iter=10; #4005;
seq=ITTIV;
task=mm_ITV_CL;
n_gpus=1;
all_seed=(1029 47 816 21 219 222 628 845 17 531 635);
dropout1=0.5;
dropout2=0.25;

# i=0

ewc_lambda=0.5

run1() {
	cuda_device=$1
	for i in 0 1
	do
		CUDA_VISIBLE_DEVICES=${cuda_device} python train_cl.py ${n_iter} ${task} ${bs} --n_gpus ${n_gpus} \
		--save_interval ${save_intvl} --eval_interval ${eval_intvl} --save_best \
		--model_save_path ${out_path}/mmcl/ITTIV_all/random_seq/seq_count/repeat${n_iter}_ewc_all_old_tasks_max5000/mmcl_random_seq_${seq}_${ewc_lambda}ewc_lambda_${bs}bs_${n_iter}it_${eval_intvl}eval_intvl_$((all_seed[$i]))_dropout_${dropout1}_${dropout2} \
		--sample_idx_fn sample_idx_cl_ITTIV_pT.7_no_dup_all_img_text_video_seed10 \
		--peripherals_type timeline --conf_type timeline \
		--all_seed $((all_seed[$i])) --rewarding --ewc_lambda ${ewc_lambda} --full_seq ${seq} > /mmcl_random_seq_${seq}_count_repeat${n_iter}_ewc_all_old_tasks_max5000_${ewc_lambda}ewc_lambda_${bs}bs_${n_iter}it_${eval_intvl}eval_intvl_$((all_seed[$i]))_dropout_${dropout1}_${dropout2}.log 2> /mmcl_random_seq_${seq}_count_repeat${n_iter}_ewc_all_old_tasks_max5000_${ewc_lambda}ewc_lambda_${bs}bs_${n_iter}it_${eval_intvl}eval_intvl_$((all_seed[$i]))_dropout_${dropout1}_${dropout2}.err;
	done
}

run2() {
	cuda_device=$1
	for i in 2 4 # 0.6 0.7
	do
		CUDA_VISIBLE_DEVICES=${cuda_device} python train_cl.py ${n_iter} ${task} ${bs} --n_gpus ${n_gpus} \
		--save_interval ${save_intvl} --eval_interval ${eval_intvl} --save_best \
		--model_save_path ${out_path}/mmcl/ITTIV_all/random_seq/seq_count/repeat${n_iter}_ewc_all_old_tasks_max5000/mmcl_random_seq_${seq}_${ewc_lambda}ewc_lambda_${bs}bs_${n_iter}it_${eval_intvl}eval_intvl_$((all_seed[$i]))_dropout_${dropout1}_${dropout2} \
		--sample_idx_fn sample_idx_cl_ITTIV_pT.7_no_dup_all_img_text_video_seed10 \
		--peripherals_type timeline --conf_type timeline \
		--all_seed $((all_seed[$i])) --rewarding --ewc_lambda ${ewc_lambda} --full_seq ${seq} > /mmcl_random_seq_${seq}_count_repeat${n_iter}_ewc_all_old_tasks_max5000_${ewc_lambda}ewc_lambda_${bs}bs_${n_iter}it_${eval_intvl}eval_intvl_$((all_seed[$i]))_dropout_${dropout1}_${dropout2}.log 2> /mmcl_random_seq_${seq}_count_repeat${n_iter}_ewc_all_old_tasks_max5000_${ewc_lambda}ewc_lambda_${bs}bs_${n_iter}it_${eval_intvl}eval_intvl_$((all_seed[$i]))_dropout_${dropout1}_${dropout2}.err;
	done
}

run3() {
	cuda_device=$1
	for i in 3 #0.8 0.9 1.0 1.1
	do
		CUDA_VISIBLE_DEVICES=${cuda_device} python train_cl.py ${n_iter} ${task} ${bs} --n_gpus ${n_gpus} \
		--save_interval ${save_intvl} --eval_interval ${eval_intvl} --save_best \
		--model_save_path ${out_path}/mmcl/ITTIV_all/random_seq/seq_count/repeat${n_iter}_ewc_all_old_tasks_max5000/mmcl_random_seq_${seq}_${ewc_lambda}ewc_lambda_${bs}bs_${n_iter}it_${eval_intvl}eval_intvl_$((all_seed[$i]))_dropout_${dropout1}_${dropout2} \
		--sample_idx_fn sample_idx_cl_ITTIV_pT.7_no_dup_all_img_text_video_seed10 \
		--peripherals_type timeline --conf_type timeline \
		--all_seed $((all_seed[$i])) --rewarding --ewc_lambda ${ewc_lambda} --full_seq ${seq} > /mmcl_random_seq_${seq}_count_repeat${n_iter}_ewc_all_old_tasks_max5000_${ewc_lambda}ewc_lambda_${bs}bs_${n_iter}it_${eval_intvl}eval_intvl_$((all_seed[$i]))_dropout_${dropout1}_${dropout2}.log 2> /mmcl_random_seq_${seq}_count_repeat${n_iter}_ewc_all_old_tasks_max5000_${ewc_lambda}ewc_lambda_${bs}bs_${n_iter}it_${eval_intvl}eval_intvl_$((all_seed[$i]))_dropout_${dropout1}_${dropout2}.err;
	done
}

run1 3 &
run2 5 &
run3 7;