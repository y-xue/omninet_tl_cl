#!/bin/bash
#cuda_device=2;
all_seed=(1029 47 816 21 219 222 628 845 17 531 635);


run1(){
	gpu_id=$1
	n_iter=$2
	seed=$3
	data_seed=$4
	sw=$5
	lr=$6
	n_warmup=$7
	slp=$8

	sleep ${slp};
	python train_with_bayesian_opt.py --gpu_id $gpu_id \
	--out_path ITTIV_all/mm_random_seq_dropout_0.5_0.25_lr${lr}_warmup${n_warmup}_allseed${seed}_dataseed${data_seed}_iter${n_iter}_sw1 \
	--seed $seed --data_seed $data_seed \
	--init_lr ${lr} --n_warmup_steps ${n_warmup} \
	--bo_n_iter 0 --n_iter ${n_iter} --sample_weights $sw < /dev/null > /dev/null 2>&1;

}

sw=1.0,1.0,1.0,1.0,1.0

run1 0 50005 1029 1029 $sw 0.02 5000 1 &
run1 1 50005 1029 1029 $sw 0.01 5000 5 &
run1 2 50005 1029 1029 $sw 0.01 16000 10 &
run1 3 50005 1029 1029 $sw 0.005 16000 15
# run1 2 54005 21 $sw &
# run1 5 54005 816 $sw &

# for s in 47 816 21 219 #1029
# do
# 	run1 3 50005 $s $s $sw
# done


# run2 1 &
# run3 2 &
# run4 3;