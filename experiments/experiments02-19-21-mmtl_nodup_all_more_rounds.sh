#!/bin/bash
#cuda_device=2;
all_seed=(1029 47 816 21 219 222 628 845 17 531 635);
# n_iter=54005

# run1(){
# 	gpu_id=$1
# 	seed=$2

# 	python train_with_bayesian_opt.py --gpu_id $gpu_id \
# 	--out_path ITTIV_all/mm_random_seq_dropout_0.5_0.25_allseed${seed}_iter${n_iter} \
# 	--seed $seed --bo_n_iter 0 --n_iter ${n_iter} < /dev/null > /dev/null 2>&1;

# }

run1(){
	gpu_id=$1
	n_iter=$2
	seed=$3
	sw=$4

	python train_with_bayesian_opt.py --gpu_id $gpu_id \
	--out_path ITTIV_all/mm_random_seq_dropout_0.5_0.25_allseed${seed}_iter${n_iter}_sw1 \
	--seed $seed --bo_n_iter 0 --n_iter ${n_iter} --sample_weights $sw < /dev/null > /dev/null 2>&1;

}

# run2(){
# 	gpu_id=$1
	
# 	for seed in 816
# 	do
# 		python train_with_bayesian_opt.py --gpu_id $gpu_id \
# 		--out_path ITTIV_all/mm_random_seq_dropout_0.5_0.25_allseed$seed \
# 		--seed $seed --bo_n_iter 0 --n_iter 27005 < /dev/null > /dev/null 2>&1;
# 	done

# }

# run3(){
# 	gpu_id=$1
	
# 	for seed in 21
# 	do
# 		python train_with_bayesian_opt.py --gpu_id $gpu_id \
# 		--out_path ITTIV_all/mm_random_seq_dropout_0.5_0.25_allseed$seed \
# 		--seed $seed --bo_n_iter 0 --n_iter 27005 < /dev/null > /dev/null 2>&1;
# 	done

# }

# run4(){
# 	gpu_id=$1
	
# 	for seed in 219
# 	do
# 		python train_with_bayesian_opt.py --gpu_id $gpu_id \
# 		--out_path ITTIV_all/mm_random_seq_dropout_0.5_0.25_allseed$seed \
# 		--seed $seed --bo_n_iter 0 --n_iter 27005 --restore 13500 < /dev/null > /dev/null 2>&1;
# 	done

# }

# run1 0 47 &
# run1 1 816 &
# run1 2 21 &
# run1 3 219;

# run1 3 1029
sw=1.0,1.0,1.0,1.0,1.0

run1 0 54005 47 $sw &
run1 1 54005 219 $sw &
run1 2 54005 21 $sw &
run1 5 54005 816 $sw &
run1 7 54005 1029 $sw


# run2 1 &
# run3 2 &
# run4 3;