#!/bin/bash
run_bo(){
	seed=$1
	gpu_id=$2
	lr=$3
	warmup=$4
	port=$5
	sleep $6;
	odir=/mnt-gluster/data/yxue/allstate/out/mm_bo
	opath=socialiq/bohb_more_budgets_15k-45k/lr${lr}_warmup${warmup}_allseed${seed}

	python train_with_bohb.py --port=$port --gpu_id $gpu_id \
	--out_path ${opath} \
	--data_path '/files/yxue/research/allstate/data/socialiq' \
	--seq QATV --task socialiq \
	--save_intvl 3899 --eval_intvl 3899 \
	--batch_size 32 --val_batch_size 2 \
	--peripherals_type='default' --conf_type='default' \
	--seed ${seed} --n_iterations 20 \
	--min_budget 15000 --max_budget 45000 \
	--lr ${lr} --n_warmup_steps ${warmup} < /dev/null > ${odir}/${opath}.nohup.log 2> ${odir}/${opath}.err;

}

run_bo 816 0 0.005 16000 9500 1 &
run_bo 816 1 0.006 16000 9600 30 &
run_bo 816 2 0.008 16000 9700 60;


