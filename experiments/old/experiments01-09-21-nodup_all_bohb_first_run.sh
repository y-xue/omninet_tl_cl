#!/bin/bash
#cuda_device=2;
all_seed=(1029 47 816 21 219 222 628 845 17 531 635);

run_bo(){
	seed=$1
	gpu_id=$2
	port=$3

	python train_with_bohb.py --port=$port --gpu_id $gpu_id \
	--out_path ITTIV_all/bohb/mm_random_seq_dropout_0.5_0.25_allseed${seed}_20bohb_iter \
	--seed ${seed} --n_iterations 20 \
	--min_budget 1000 --max_budget 4000; # < /dev/null > /dev/null 2>&1;

}

# run_bo 1029 0 9100 &
# run_bo 47 1 9200 &
# run_bo 219 2 9300 ;

run_bo 816 1 9400 &
run_bo 21 2 9500;