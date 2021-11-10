#!/bin/bash
#cuda_device=2;
all_seed=(1029 47 816 21 219 222 628 845 17 531 635);


run1(){
	gpu_id=$1
	n_iter=$2
	seed=$3
	data_seed=$4
	sw=$5
	slp=$6

	out_path=mm_random_seq_dropout_0.5_0.25_allseed${seed}_dataseed${data_seed}_iter${n_iter}_sw1

	sleep ${slp};
	python train_with_bayesian_opt.py --gpu_id $gpu_id \
	--out_path ITTIV_all/${out_path} \
	--seed $seed --data_seed $data_seed \
	--bo_n_iter 0 --n_iter ${n_iter} --sample_weights $sw --skip_read_best > /${out_path}.log 2> /${out_path}.err;

}

sw=1.0,1.0,1.0,1.0,1.0

# run1 0 300005 816 816 $sw 1 &
run1 0 300005 21 21 $sw 1