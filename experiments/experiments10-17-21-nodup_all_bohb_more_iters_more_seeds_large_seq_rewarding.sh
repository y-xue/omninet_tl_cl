#!/bin/bash
run_bo(){
	seed=$1
	gpu_id=$2
	port=$3
	sleep $4;
	odir=/files/yxue/research/allstate/out/mm_bo
	opath=ITTIV_all/bohb_more_budgets_15k-50k/large_seq_rewarding/mm_random_seq_dropout_0.5_0.25_allseed${seed}_20bohb_iter

	python train_with_bohb.py --port=$port --gpu_id $gpu_id \
	--out_path ${opath} \
	--seed ${seed} --n_iterations 20 \
	--min_budget 15000 --max_budget 50000 \
	--lr 0.02 --n_warmup_steps 16000 --large_seq_rewarding < /dev/null > ${odir}/${opath}.nohup.log 2> ${odir}/${opath}.err;

}

run_bo 2 0 9600 1 &
run_bo 948 1 9700 60;

