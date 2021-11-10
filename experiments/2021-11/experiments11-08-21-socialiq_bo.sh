#!/bin/bash
run_bo(){
	seed=$1
	gpu_id=$2
	lr=$3
	warmup=$4
	port=$5
	sleep $6;
	odir=/files/yxue/research/allstate/out/mm_bo
	opath=socialiq/bohb_more_budgets_15k-50k/lr${lr}_warmup${warmup}_allseed${seed}

	python train_with_bohb.py --port=$port --gpu_id $gpu_id \
	--out_path ${opath} \
	--data_path '/files/yxue/research/allstate/data/socialiq' \
	--seq QATV --task socialiq \
	--save_intvl 3899 --eval_intvl 3899 \
	--batch_size 16 --val_batch_size 4 \
	--peripherals_type='default' --conf_type='default' \
	--seed ${seed} --n_iterations 20 \
	--min_budget 15000 --max_budget 50000 \
	--lr ${lr} --n_warmup_steps ${warmup} < /dev/null > ${odir}/${opath}.nohup.log 2> ${odir}/${opath}.err;

}

run_bo 21 0 0.005 16000 9500 1
