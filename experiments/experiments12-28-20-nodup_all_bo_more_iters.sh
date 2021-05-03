#!/bin/bash
#cuda_device=2;
all_seed=(1029 47 816 21 219 222 628 845 17 531 635);

run_bo(){
	seed=$1
	gpu_id=$2

	python train_with_bayesian_opt.py --gpu_id $gpu_id \
	--out_path ITTIV_all/mm_random_seq_dropout_0.5_0.25_upper_bound_1_random_3_points_allseed$seed \
	--seed $seed --n_iter 30 < /dev/null > /dev/null 2>&1;

}

run_bo 1029 1 &
run_bo 47 3;