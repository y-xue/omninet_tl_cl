#!/bin/bash
run_bo(){
	seed=$1
	gpu_id=$2
	port=$3

	python train_with_bohb.py --port=$port --gpu_id $gpu_id \
	--out_path ITTIV_all/bohb_more_budgets_15k-50k/mm_random_seq_dropout_0.5_0.25_allseed${seed}_20bohb_iter \
	--seed ${seed} --n_iterations 20 \
	--min_budget 15000 --max_budget 50000 --lr 0.02 --n_warmup_steps 16000; # < /dev/null > /dev/null 2>&1;

}

# run_bo 219 1 9500 &
# run_bo 1029 1 9600 &
# run_bo 47 2 9700 &
# run_bo 816 3 9800;
run_bo 816 0 9800 1 &
run_bo 21 1 9400 60 &
run_bo 1029 3 9600 120 &
run_bo 219 2 9500 180;