#!/bin/bash
out_path=/files/yxue/research/allstate/out;

save_intvl=100;
eval_intvl=100;
bs=16;
n_iter=3;
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
# all_seed_i=3

run() {
	cuda_device=$1
	l1=$2
	l2=$3
	l3=$4
	l4=$5
	l5=$6
	init_lr=$7
	all_seed_i=$8
	CUDA_VISIBLE_DEVICES=${cuda_device} python train_cl.py ${n_iter} ${task} ${bs} --n_gpus ${n_gpus} \
	--save_interval ${save_intvl} --eval_interval ${eval_intvl} --save_best \
	--model_save_path ${out_path}/mmcl/ITTIV_all/random_seq/seq_count/repeat${n_iter}/ewc_all_other_tasks_${max_iter}_early_stop_cw${converge_window}_sw${safe_window}_restore_optimizer_lr/mmcl_random_seq_${seq}_${l1}_${l2}_${l3}_${l4}_${l5}ewc_lambda_lr${init_lr}_${bs}bs_${eval_intvl}eval_intvl_allseed$((all_seed[$all_seed_i]))_dataseed$((all_seed[$all_seed_i])) \
	--sample_idx_fn sample_idx_cl_ITTIV_pT.7_no_dup_all_img_text_video_seed10 \
	--peripherals_type timeline --conf_type timeline \
	--all_seed $((all_seed[$all_seed_i])) --data_seed $((all_seed[$all_seed_i])) \
	--rewarding --ewc_lambda $l1 $l2 $l3 $l4 $l5 --full_seq ${seq} \
	--converge_window ${converge_window} --safe_window ${safe_window} --restore_opt_lr \
	--init_lr $init_lr > /mmcl_random_seq_${seq}_count_repeat${n_iter}_ewc_all_other_tasks_${max_iter}_early_stop_cw${converge_window}_sw${safe_window}_restore_optimizer_lr_${l1}_${l2}_${l3}_${l4}_${l5}ewc_lambda_lr${init_lr}_${bs}bs_${n_iter}it_${eval_intvl}eval_intvl_allseed$((all_seed[$all_seed_i]))_dataseed$((all_seed[$all_seed_i])).log 2> /mmcl_random_seq_${seq}_count_repeat${n_iter}_ewc_all_other_tasks_${max_iter}_early_stop_cw${converge_window}_sw${safe_window}_restore_optimizer_lr_${l1}_${l2}_${l3}_${l4}_${l5}ewc_lambda_lr${init_lr}_${bs}bs_${n_iter}it_${eval_intvl}eval_intvl_allseed$((all_seed[$all_seed_i]))_dataseed$((all_seed[$all_seed_i])).err;
}


run1() {
	# lr=0.005
	for l5 in 150 200 250 300
	do
		run 0 0 0 0 0 ${l5} 0.005 2;
		sleep 5;
	done
}

run2() {
	# lr=0.01
	sleep 5
	for l5 in 150 200 250 300
	do
		run 1 0 0 0 0 ${l5} 0.01 2;
		sleep 5;
	done
}

run3() {
	# lr=0.008
	sleep 10
	for l5 in 150 200 250 300
	do
		run 2 0 0 0 0 ${l5} 0.008 2;
		sleep 5;
	done
}

run4() {
	# lr=0.003
	sleep 15
	for l5 in 150 200 250 300
	do
		run 3 0 0 0 0 ${l5} 0.003 2;
		sleep 5;
	done
}


run1 &
run2 &
run3 &
run4

