#!/bin/bash

run() {
    opath=/files/yxue/research/allstate/out/socialiq_cl/lr0.004_warmup5000_ep15_exp3-10-1-1-1_seed1029_task3-bs20_0.25eval_interval
    CUDA_VISIBLE_DEVICES=3 python train_cl.py 5 socialiq 32 \
    --val_batch_size 2 --n_gpus 1 --n_workers 0 \
    --data_path /files/yxue/research/allstate/data --save_best \
    --model_save_path ${opath} \
    --all_seed 1029 --data_seed 1029 --n_warmup_steps 5000 --rewarding --full_seq QATV \
    --converge_window 5 --safe_window 10 --ewc_lambda 10 --lambda_decay exp3 \
    --decay_a 1 --decay_b 1 --decay_g 1 \
    --restore_opt_lr --init_lr 0.004 \
    --restore_cl_last > /dev/null 2> ${opath}.err;
    sleep 30;
}

for i in {1..20}
do
    run
done
