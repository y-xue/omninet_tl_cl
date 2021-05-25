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
	warmup1=$9
    warmup2=${10}
    warmup3=${11}
    warmup4=${12}
    warmup5=${13}

	CUDA_VISIBLE_DEVICES=${cuda_device} python train_cl.py ${n_iter} ${task} ${bs} --n_gpus ${n_gpus} \
	--save_interval ${save_intvl} --eval_interval ${eval_intvl} --save_best \
	--model_save_path ${out_path}/mmcl/ITTIV_all/random_seq/seq_count/repeat${n_iter}/ewc_all_other_tasks_${max_iter}_early_stop_cw${converge_window}_sw${safe_window}_restore_optimizer_lr_all/mmcl_random_seq_${seq}_${l1}_${l2}_${l3}_${l4}_${l5}ewc_lambda_lr${init_lr}_nwarmup${warmup1}_${warmup2}_${warmup3}_${warmup4}_${warmup5}_${bs}bs_${eval_intvl}eval_intvl_allseed$((all_seed[$all_seed_i]))_dataseed$((all_seed[$all_seed_i])) \
	--sample_idx_fn sample_idx_cl_ITTIV_pT.7_no_dup_all_img_text_video_seed10 \
	--peripherals_type timeline --conf_type timeline \
	--all_seed $((all_seed[$all_seed_i])) --data_seed $((all_seed[$all_seed_i])) \
	--rewarding --ewc_lambda $l1 $l2 $l3 $l4 $l5 --full_seq ${seq} \
	--converge_window ${converge_window} --safe_window ${safe_window} \
	--n_warmup_steps $warmup1 $warmup2 $warmup3 $warmup4 $warmup5 \
	--restore_opt_lr_all --init_lr $init_lr > /mmcl_random_seq_${seq}_count_repeat${n_iter}_ewc_all_other_tasks_${max_iter}_early_stop_cw${converge_window}_sw${safe_window}_restore_optimizer_lr_all_${l1}_${l2}_${l3}_${l4}_${l5}ewc_lambda_lr${init_lr}_nwarmup${warmup1}_${warmup2}_${warmup3}_${warmup4}_${warmup5}_${bs}bs_${n_iter}it_${eval_intvl}eval_intvl_allseed$((all_seed[$all_seed_i]))_dataseed$((all_seed[$all_seed_i])).log 2> /mmcl_random_seq_${seq}_count_repeat${n_iter}_ewc_all_other_tasks_${max_iter}_early_stop_cw${converge_window}_sw${safe_window}_restore_optimizer_lr_all_${l1}_${l2}_${l3}_${l4}_${l5}ewc_lambda_lr${init_lr}_nwarmup${warmup1}_${warmup2}_${warmup3}_${warmup4}_${warmup5}_${bs}bs_${n_iter}it_${eval_intvl}eval_intvl_allseed$((all_seed[$all_seed_i]))_dataseed$((all_seed[$all_seed_i])).err;
}

run1() {
	for lr in 0.008 0.01 
	do
		run 0 20 0 0 50 100 ${lr} 2 2000 2000 2000 2000 200;
		sleep 5;
	done
}

run2() {
	sleep 5
	for lr in 0.003 0.005
	do
		run 1 20 0 0 50 100 ${lr} 2 2000 2000 2000 2000 200;
		sleep 5;
	done
}

run3() {
	sleep 10
	for lr in 0.02 0.002
	do
		run 3 20 0 0 50 100 ${lr} 2 2000 2000 2000 2000 200;
		sleep 5;
	done
}


run1 &
run2 &
run3

