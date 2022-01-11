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
    seed=${11}
    sleep ${12};

    for i in {1..20}
    do
        opath=/files/yxue/research/allstate/out/socialiq_cl/lr${lr}_warmup5000_ep15_exp3-${l}-${a}-${b}-${g}_seed${seed}_task3-bs32_0.25eval_interval_es${es1}-${es2}-${es3}-${es4}
        CUDA_VISIBLE_DEVICES=${gpu_id} python train_cl.py 5 socialiq 32 \
        --val_batch_size 2 --n_gpus 1 --n_workers 2 \
        --data_path /files/yxue/research/allstate/data --save_best \
        --model_save_path ${opath} \
        --all_seed ${seed} --data_seed ${seed} --n_warmup_steps 5000 --rewarding --full_seq QATV \
        --converge_window 10 --safe_window 80 --ewc_lambda ${l} --lambda_decay exp3 \
        --decay_a ${a} --decay_b ${b} --decay_g ${g} \
        --subtask_epochs ${es1} ${es2} ${es3} ${es4} \
        --restore_opt_lr --init_lr ${lr} \
        --restore_cl_last > /dev/null 2> ${opath}.err;
        sleep 30;
    done
}


run 0 0.7 0.7 0.5 0.003 30 30 30 10 10 47 1 &
run 1 0.7 0.7 0.5 0.003 30 30 30 10 10 816 30 &
run 2 0.7 0.7 0.5 0.003 30 30 30 10 10 722 60 &
run 3 0.7 0.7 0.5 0.003 30 30 30 10 10 219 90;

