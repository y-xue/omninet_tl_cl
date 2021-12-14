#!/bin/bash
run(){
	seed=$1
	gpu_id=$2
	lr=$3
	warmup=$4

	opath=/files/yxue/research/allstate/out/socialiq_tl_cl/omninet_bs32_lr${lr}_nwarmup${warmup}_seed${seed}; 
	CUDA_VISIBLE_DEVICES=${gpu_id} python train.py 150000 socialiq 32 \
	--n_gpus 1 --save_interval 3899 --eval_interval 3899 \
	--init_lr ${lr} --n_warmup_steps ${warmup} --save_best --model_save_path ${opath} \
	--move_out_path ${opath} --all_seed ${seed} --rewarding \
	--restore_last --val_batch_size 2 < /dev/null > /dev/null 2> ${opath}.err;

	sleep 30;

	rm -r ${opath};
}

for s in 816 47 21 219
do
	run $s 2 0.005 16000;
	run $s 2 0.004 16000;
done


