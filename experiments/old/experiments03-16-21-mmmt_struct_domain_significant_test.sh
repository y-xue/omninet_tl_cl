# CUDA_VISIBLE_DEVICES=0 nohup python train.py 841110 vqa_struct 64 --n_gpus 1 \
# --save_interval 6470 --eval_interval 6470 --conf_type vqa_struct \
# --model_save_path /files/yxue/research/allstate/out/omninet/vqa_struct_norm_bs64_struct_periph_3_encoder_only_fusion_gating_per_frame_softmax_convex_dp0.1_0.5_0.5 \
# --move_out_path /files/yxue/research/allstate/out/omninet/vqa_struct_norm_bs64_struct_periph_3_encoder_only_fusion_gating_per_frame_softmax_convex_dp0.1_0.5_0.5 \
# --structured_folder synthetic_structured_clustering_std3 --inject_at_encoder \
# --temp_fusion_attn_type selective --spat_fusion_attn_type selective \
# --struct_temp_periph_dropout 0.5 --struct_spat_periph_dropout 0.5 \
# --convex_gate --n_workers 2 --fusion < /dev/null > /dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python train.py 841110 vqa_struct 64 --n_gpus 1 \
# --save_interval 6470 --eval_interval 6470 --conf_type vqa_struct \
# --model_save_path /files/yxue/research/allstate/out/omninet/vqa_struct_norm_bs64_struct_periph_3_encoder_only_fusion_gating_per_frame_softmax_convex_dp0.1_0.5_0.5_lr0.01 \
# --move_out_path /files/yxue/research/allstate/out/omninet/vqa_struct_norm_bs64_struct_periph_3_encoder_only_fusion_gating_per_frame_softmax_convex_dp0.1_0.5_0.5_lr0.01 \
# --structured_folder synthetic_structured_clustering_std3 --inject_at_encoder \
# --temp_fusion_attn_type selective --spat_fusion_attn_type selective \
# --struct_temp_periph_dropout 0.5 --struct_spat_periph_dropout 0.5 \
# --convex_gate --n_workers 2 --fusion --init_lr 0.01 < /dev/null > /dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python train.py 841110 vqa_struct 64 --n_gpus 1 \
# --save_interval 6470 --eval_interval 6470 --conf_type vqa_struct \
# --model_save_path /files/yxue/research/allstate/out/omninet/struct_domain/vqa_struct_norm_bs64_struct_periph_3_encoder_only_dp0.1_0.8_0.8_init_lr0.01 \
# --move_out_path /files/yxue/research/allstate/out/omninet/struct_domain/vqa_struct_norm_bs64_struct_periph_3_encoder_only_dp0.1_0.8_0.8_init_lr0.01 \
# --structured_folder synthetic_structured_clustering_std3 --inject_at_encoder \
# --n_workers 2 --init_lr 0.01 --struct_periph_dropout 0.1 --struct_temp_periph_dropout 0.8 \
# --struct_spat_periph_dropout 0.8 --restore_last < /dev/null > /dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python train.py 841110 vqa_struct 64 --n_gpus 1 \
# --save_interval 6470 --eval_interval 6470 --conf_type vqa_struct \
# --model_save_path /files/yxue/research/allstate/out/omninet/struct_domain/vqa_struct_norm_bs64_struct_periph_3_encoder_only_dp0.1_0.3_0.3_init_lr0.01 \
# --move_out_path /files/yxue/research/allstate/out/omninet/struct_domain/vqa_struct_norm_bs64_struct_periph_3_encoder_only_dp0.1_0.3_0.3_init_lr0.01 \
# --structured_folder synthetic_structured_clustering_std3 --inject_at_encoder \
# --n_workers 2 --init_lr 0.01 --struct_periph_dropout 0.1 --struct_temp_periph_dropout 0.3 \
# --struct_spat_periph_dropout 0.3 < /dev/null > /dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python train.py 841110 vqa_struct 64 --n_gpus 1 \
# --save_interval 6470 --eval_interval 6470 --conf_type vqa_struct \
# --model_save_path /files/yxue/research/allstate/out/omninet/struct_domain/vqa_struct_norm_bs64_struct_periph_3_encoder_only_dp0.1_0.1_0.1_init_lr0.01 \
# --move_out_path /files/yxue/research/allstate/out/omninet/struct_domain/vqa_struct_norm_bs64_struct_periph_3_encoder_only_dp0.1_0.1_0.1_init_lr0.01 \
# --structured_folder synthetic_structured_clustering_std3 --inject_at_encoder \
# --n_workers 2 --init_lr 0.01 --struct_periph_dropout 0.1 --struct_temp_periph_dropout 0.1 \
# --struct_spat_periph_dropout 0.1 < /dev/null > /dev/null 2>&1 &


# CUDA_VISIBLE_DEVICES=2 nohup python train.py 841110 vqa_struct 64 --n_gpus 1 \
# --save_interval 6470 --eval_interval 6470 --conf_type vqa_struct \
# --model_save_path /files/yxue/research/allstate/out/omninet/struct_domain/vqa_struct_norm_bs64_struct_periph_3_encoder_only_dp0.1_0.5_0.5_init_lr0.02 \
# --move_out_path /files/yxue/research/allstate/out/omninet/struct_domain/vqa_struct_norm_bs64_struct_periph_3_encoder_only_dp0.1_0.5_0.5_init_lr0.02 \
# --structured_folder synthetic_structured_clustering_std3 --inject_at_encoder \
# --n_workers 2 --init_lr 0.02 --struct_periph_dropout 0.1 --struct_temp_periph_dropout 0.3 \
# --struct_spat_periph_dropout 0.3 < /dev/null > /dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=3 nohup python train.py 841110 vqa_struct 64 --n_gpus 1 \
# --save_interval 6470 --eval_interval 6470 --conf_type vqa_struct \
# --model_save_path /files/yxue/research/allstate/out/omninet/struct_domain/vqa_struct_norm_bs64_struct_periph_3_encoder_only_dp0.1_0.5_0.5_init_lr0.005 \
# --move_out_path /files/yxue/research/allstate/out/omninet/struct_domain/vqa_struct_norm_bs64_struct_periph_3_encoder_only_dp0.1_0.5_0.5_init_lr0.005 \
# --structured_folder synthetic_structured_clustering_std3 --inject_at_encoder \
# --n_workers 2 --init_lr 0.005 --struct_periph_dropout 0.1 --struct_temp_periph_dropout 0.3 \
# --struct_spat_periph_dropout 0.3 < /dev/null > /dev/null 2>&1 &

seed=1001;
out_path=/scratch1/yxuea/out/omninet/struct_domain/vqa_struct_norm_bs64_struct_periph_3_encoder_only_dp0.1_0.5_0.5_init_lr0.01_allseed${seed}
CUDA_VISIBLE_DEVICES=3 nohup python train.py 841110 vqa_struct 64 --n_gpus 1 \
--save_interval 6470 --eval_interval 6470 --conf_type vqa_struct \
--data_path /scratch1/yxuea/data \
--model_save_path ${out_path} \
--move_out_path ${out_path} \
--structured_folder synthetic_structured_clustering_std3 --inject_at_encoder \
--n_workers 2 --init_lr 0.01 --struct_periph_dropout 0.1 --struct_temp_periph_dropout 0.5 \
--struct_spat_periph_dropout 0.5 --all_seed ${seed} < /dev/null > /dev/null 2> ${out_path}.err &

