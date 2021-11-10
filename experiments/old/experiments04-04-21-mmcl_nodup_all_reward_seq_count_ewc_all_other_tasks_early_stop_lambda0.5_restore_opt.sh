#!/bin/bash
out_path=/files/yxue/research/allstate/out;

save_intvl=100;
eval_intvl=100;
bs=16;
n_iter=10;
seq=ITTIV;
task=mm_ITV_CL;
n_gpus=1;
all_seed=(1029 47 816 21 219 222 628 845 17 531 635);
dropout1=0.5;
dropout2=0.25;

# i=0

converge_window=10
safe_window=30

max_iter=10000
ewc_lambda_str=0.5
i=3

run() {
	cuda_device=$1

	CUDA_VISIBLE_DEVICES=${cuda_device} python train_cl.py ${n_iter} ${task} ${bs} --n_gpus ${n_gpus} \
	--save_interval ${save_intvl} --eval_interval ${eval_intvl} --save_best \
	--model_save_path ${out_path}/mmcl/ITTIV_all/random_seq/seq_count/repeat${n_iter}/ewc_all_other_tasks_${max_iter}_early_stop_cw${converge_window}_sw${safe_window}_restore_optimizer/mmcl_random_seq_${seq}_${ewc_lambda_str}ewc_lambda_${bs}bs_${n_iter}it_${eval_intvl}eval_intvl_$((all_seed[$i]))_dropout_${dropout1}_${dropout2} \
	--sample_idx_fn sample_idx_cl_ITTIV_pT.7_no_dup_all_img_text_video_seed10 \
	--peripherals_type timeline --conf_type timeline \
	--all_seed $((all_seed[$i])) --rewarding --ewc_lambda 0.5 --full_seq ${seq} \
	--converge_window ${converge_window} --safe_window ${safe_window} --restore_opt > /mmcl_random_seq_${seq}_count_repeat${n_iter}_ewc_all_other_tasks_${max_iter}_early_stop_cw${converge_window}_sw${safe_window}_restore_optimizer_${ewc_lambda_str}ewc_lambda_${bs}bs_${n_iter}it_${eval_intvl}eval_intvl_$((all_seed[$i]))_dropout_${dropout1}_${dropout2}.log 2> /mmcl_random_seq_${seq}_count_repeat${n_iter}_ewc_all_other_tasks_${max_iter}_early_stop_cw${converge_window}_sw${safe_window}_restore_optimizer_${ewc_lambda_str}ewc_lambda_${bs}bs_${n_iter}it_${eval_intvl}eval_intvl_$((all_seed[$i]))_dropout_${dropout1}_${dropout2}.err;
}

run 0