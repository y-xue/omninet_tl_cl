#!/bin/bash
#cuda_device=2;
all_seed=(1029 47 816 21 219 222 628 845 17 531 635);

# run1(){
# 	for i in 1 2 #{0..4}
# 	do
# 		python train_with_bayesian_opt.py --gpu_id 2 \
# 		--out_path ITTIV_all/mm_random_seq_dropout_0.5_0.25_upper_bound_1_random_3_points_allseed$((all_seed[$i])) \
# 		--seed $((all_seed[$i])) --n_iter 15 < /dev/null > /dev/null 2>&1;
# 		sleep 10;
# 	done
# }

# run2(){
# 	for i in 3 4 #{0..4}
# 	do
# 		python train_with_bayesian_opt.py --gpu_id 3 \
# 		--out_path ITTIV_all/mm_random_seq_dropout_0.5_0.25_upper_bound_1_random_3_points_allseed$((all_seed[$i])) \
# 		--seed $((all_seed[$i])) --n_iter 15 < /dev/null > /dev/null 2>&1;
# 		sleep 10;
# 	done
# }

# run1 &
# run2

# i=0;
# python train_with_bayesian_opt.py --gpu_id 1 \
# --out_path ITTIV_all/mm_random_seq_dropout_0.5_0.25_upper_bound_1_random_3_points_allseed$((all_seed[$i])) \
# --seed $((all_seed[$i])) --n_iter 15 --test_bo_iter -4 < /dev/null > /dev/null 2>&1;
# sleep 10;
# python train_with_bayesian_opt.py --gpu_id 1 \
# --out_path ITTIV_all/mm_random_seq_dropout_0.5_0.25_upper_bound_1_random_3_points_allseed$((all_seed[$i])) \
# --seed $((all_seed[$i])) --n_iter 15 --test_bo_iter 14 < /dev/null > /dev/null 2>&1;
# sleep 10;

# i=1;
# python train_with_bayesian_opt.py --gpu_id 1 \
# --out_path ITTIV_all/mm_random_seq_dropout_0.5_0.25_upper_bound_1_random_3_points_allseed$((all_seed[$i])) \
# --seed $((all_seed[$i])) --n_iter 15 --test_bo_iter -4 < /dev/null > /dev/null 2>&1;
# sleep 10;
# python train_with_bayesian_opt.py --gpu_id 1 \
# --out_path ITTIV_all/mm_random_seq_dropout_0.5_0.25_upper_bound_1_random_3_points_allseed$((all_seed[$i])) \
# --seed $((all_seed[$i])) --n_iter 15 --test_bo_iter 6 < /dev/null > /dev/null 2>&1;
# sleep 10;

# i=3;
# python train_with_bayesian_opt.py --gpu_id 1 \
# --out_path ITTIV_all/mm_random_seq_dropout_0.5_0.25_upper_bound_1_random_3_points_allseed$((all_seed[$i])) \
# --seed $((all_seed[$i])) --n_iter 15 --test_bo_iter -4 < /dev/null > /dev/null 2>&1;
# sleep 10;
# python train_with_bayesian_opt.py --gpu_id 1 \
# --out_path ITTIV_all/mm_random_seq_dropout_0.5_0.25_upper_bound_1_random_3_points_allseed$((all_seed[$i])) \
# --seed $((all_seed[$i])) --n_iter 15 --test_bo_iter -1 < /dev/null > /dev/null 2>&1;
# sleep 10;


# i=2
# python train_with_bayesian_opt.py --gpu_id 2 \
# --out_path ITTIV_all/mm_random_seq_dropout_0.5_0.25_upper_bound_1_random_3_points_allseed$((all_seed[$i])) \
# --seed $((all_seed[$i])) --n_iter 30 < /dev/null > /dev/null 2>&1;

i=4;
python train_with_bayesian_opt.py --gpu_id 2 \
--out_path ITTIV_all/mm_random_seq_dropout_0.5_0.25_upper_bound_1_random_3_points_allseed$((all_seed[$i])) \
--seed $((all_seed[$i])) --n_iter 15 --test_bo_iter -4 < /dev/null > /dev/null 2>&1;
sleep 10;
python train_with_bayesian_opt.py --gpu_id 2 \
--out_path ITTIV_all/mm_random_seq_dropout_0.5_0.25_upper_bound_1_random_3_points_allseed$((all_seed[$i])) \
--seed $((all_seed[$i])) --n_iter 15 --test_bo_iter -3 < /dev/null > /dev/null 2>&1;
sleep 10;

