#!/bin/bash
run(){
	seed=$1
	gpu_id=$2
	lr=$3
	warmup=$4
	full_seq=$5

	opath=/files/yxue/research/allstate/out/socialiq_tl_cl/omninet_bs32_${full_seq}_lr${lr}_nwarmup${warmup}_seed${seed}; 
	CUDA_VISIBLE_DEVICES=${gpu_id} python train.py 150000 socialiq 32 \
	--n_gpus 1 --save_interval 1072 --eval_interval 1072 \
	--init_lr ${lr} --n_warmup_steps ${warmup} --save_best --model_save_path ${opath} \
	--move_out_path ${opath} --all_seed ${seed} --rewarding \
	--full_seq ${full_seq} \
	--restore_last --val_batch_size 2 < /dev/null > /dev/null 2> ${opath}.err;

	sleep 30;

	rm -r ${opath};
}

for lr in 0.005 0.004 0.003 0.002
do
	for warmp in 16000 5000
	do
		run 1029 0 ${lr} ${warmp} Q;
	done
done


