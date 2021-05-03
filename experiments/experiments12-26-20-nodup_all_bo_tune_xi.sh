#!/bin/bash
#cuda_device=2;
all_seed=(1029 47 816 21 219 222 628 845 17 531 635);

run_bo(){
	seed=$1
	xi=$2
	gpu_id=$3

	python train_with_bayesian_opt.py --gpu_id $gpu_id \
	--out_path ITTIV_all/mm_random_seq_dropout_0.5_0.25_upper_bound_1_random_3_points_allseed$seed/xi$xi \
	--seed $seed --n_iter 20 --xi $xi < /dev/null > /dev/null 2>&1;

}

run_bo 816 0.3 0 &
run_bo 816 0.001 2

