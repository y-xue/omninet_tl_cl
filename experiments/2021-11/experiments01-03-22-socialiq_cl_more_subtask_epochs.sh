#!/bin/bash

run() {
    gpu_id=$1;
    a=$2;
    b=$3;
    g=$4;
    lr=$5;
    es1=$6;
    es2=$7;
    es3=$8;
    es4=$9;
    l=${10};
    warmup=${11};
    sleep ${12};

    for i in {1..20}
    do
        opath=/files/yxue/research/allstate/out/socialiq_cl/lr${lr}_warmup${warmup}_ep15_exp3-${l}-${a}-${b}-${g}_seed1029_task3-bs20_0.25eval_interval_es${es1}-${es2}-${es3}-${es4}
        CUDA_VISIBLE_DEVICES=${gpu_id} python train_cl.py 5 socialiq 32 \
        --val_batch_size 2 --n_gpus 1 --n_workers 2 \
        --data_path /files/yxue/research/allstate/data --save_best \
        --model_save_path ${opath} \
        --all_seed 1029 --data_seed 1029 --n_warmup_steps ${warmup} --rewarding --full_seq QATV \
        --converge_window 5 --safe_window 80 --ewc_lambda ${l} --lambda_decay exp3 \
        --decay_a ${a} --decay_b ${b} --decay_g ${g} \
        --subtask_epochs ${es1} ${es2} ${es3} ${es4} \
        --restore_opt_lr --init_lr ${lr} \
        --restore_cl_last > /dev/null 2> ${opath}.err;
        sleep 30;
    done
}

run 0 0.7 0.7 0.5 0.002 30 15 5 5 10 5000 1 &
run 1 0.7 0.7 0.5 0.003 30 15 5 5 10 5000 30 &
run 2 0.7 0.7 0.5 0.004 30 15 5 5 10 5000 60 &
run 3 0.7 0.7 0.5 0.003 30 15 5 5 10 16000 90;
# run 0 0.7 0.7 0.5 0.004 4 4 4 3 90;