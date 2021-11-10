#!/bin/bash
#cuda_device=2;
all_seed=(1029 47 816 21 219 222 628 845 17 531 635);


run1(){
	gpu_id=$1
	n_iter=$2
	seed=$3
	data_seed=$4
	sw=$5

	python train_with_bayesian_opt.py --gpu_id $gpu_id \
	--out_path ITTIV_all/mm_random_seq_dropout_0.5_0.25_allseed${seed}_dataseed${data_seed}_iter${n_iter}_sw1 \
	--seed $seed --data_seed $data_seed --bo_n_iter 0 --n_iter ${n_iter} --sample_weights $sw < /dev/null > /dev/null 2>&1;

}

sw=1.0,1.0,1.0,1.0,1.0

# run1 0 54005 47 $sw &
# run1 1 54005 219 $sw &
# run1 2 54005 21 $sw &
# run1 5 54005 816 $sw &

for s in 47 816 21 219 #1029
do
	run1 3 50005 $s $s $sw
done


# run2 1 &
# run3 2 &
# run4 3;