#!/bin/bash
#cuda_device=2;
all_seed=(1029 47 816 21 219 222 628 845 17 531 635);
# n_iter=54005

run1(){
	gpu_id=$1
	n_iter=$2
	seed=$3
	data_seed=$4
	sw=$5

	python train_with_bayesian_opt.py --gpu_id $gpu_id \
	--out_path ITTIV_all/mm_random_seq_dropout_0.5_0.25_allseed${seed}_iter${n_iter}_sw_best_bohb_seed${seed}_dataseed${data_seed} \
	--seed $seed --data_seed $data_seed --bo_n_iter 0 --n_iter ${n_iter} --sample_weights $sw < /dev/null > /dev/null 2>&1;

}

# run_restore(){
# 	gpu_id=$1
# 	n_iter=$2
# 	seed=$4
# 	sw=$5
# 	python train_with_bayesian_opt.py --gpu_id $gpu_id \
# 	--out_path ITTIV_all/mm_random_seq_dropout_0.5_0.25_allseed${seed}_iter${n_iter}_best_bo4000 \
# 	--seed $seed --bo_n_iter 0 --n_iter ${n_iter} --sample_weights $sw --restore $3 < /dev/null > /dev/null 2>&1;
# }

for ds in 1029 816 21 219
do
	run1 0 54005 47 $ds 0.11348847189364952,0.9744830944364566,0.7287346335011062,0.35146780589270143,0.707605138259081
done

