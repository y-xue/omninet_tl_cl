#!/bin/bash
#cuda_device=2;
all_seed=(1029 47 816 21 219 222 628 845 17 531 635);
# n_iter=54005

run1(){
	gpu_id=$1
	n_iter=$2
	seed=$3
	sw=$4

	python train_with_bayesian_opt.py --gpu_id $gpu_id \
	--out_path ITTIV_all/mm_random_seq_dropout_0.5_0.25_allseed${seed}_iter${n_iter}_sw_best_seed219 \
	--seed $seed --bo_n_iter 0 --n_iter ${n_iter} --sample_weights $sw < /dev/null > /dev/null 2>&1;

}

run_restore(){
	gpu_id=$1
	n_iter=$2
	seed=$4
	sw=$5
	python train_with_bayesian_opt.py --gpu_id $gpu_id \
	--out_path ITTIV_all/mm_random_seq_dropout_0.5_0.25_allseed${seed}_iter${n_iter}_best_bo4000 \
	--seed $seed --bo_n_iter 0 --n_iter ${n_iter} --sample_weights $sw --restore $3 < /dev/null > /dev/null 2>&1;
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

# run_restore 1 70005 54000 1029 0.00355812,0.97668259,0.37901312,0.02908752,0.50580935 &
# run1 5 54005 21 0.06957096,0.86740449,0.13324053,0.17812467,0.49592955 &
# run1 7 54005 219 0.01349341,0.27616734,0.18045594,0.27165718,0.91172968
# run1 5 54005 47 0.07042569,0.34320218,0.86325514,0.86264503,0.99617283 &
# run1 7 54005 816 0.70986571,0.81909892,0.95376684,0.01826131,0.82028076

# run1 3 70005 1029 0.00355812,0.97668259,0.37901312,0.02908752,0.50580935

run1 2 54005 21 0.01349341,0.27616734,0.18045594,0.27165718,0.91172968 &
run1 5 54005 816 0.01349341,0.27616734,0.18045594,0.27165718,0.91172968 &
run1 7 54005 1029 0.01349341,0.27616734,0.18045594,0.27165718,0.91172968

# run2 1 &
# run3 2 &
# run4 3;