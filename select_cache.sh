# my_omninet

CUDA_VISIBLE_DEVICES=2 nohup python train.py 20 vqa 128 \
--n_gpus 1 --save_interval 20 --eval_interval 20 --model_save_path /out \
--move_out_path /mnt-gluster/data/yxue/research/allstate/out/omninet/vqa_select_cache_invocab \
--restore_last --select_cache
> /vqa_select_cache_invocab.log



CUDA_VISIBLE_DEVICES=2 nohup python train.py 20 vqa 64 \
--n_gpus 1 --save_interval 20 --eval_interval 20 --model_save_path /mnt-gluster/data/yxue/research/allstate/out/omninet/vqa_select_cache_invocab \
--restore_last --select_cache
> /vqa_select_cache_invocab.log