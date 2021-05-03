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

# CUDA_VISIBLE_DEVICES=0 nohup python train.py 841110 vqa_struct 64 --n_gpus 1 \
# --save_interval 6470 --eval_interval 6470 --conf_type vqa_struct \
# --model_save_path /files/yxue/research/allstate/out/omninet/struct_domain_img_text_encoding_as_struct/vqa_struct_norm_bs64_struct_periph_3_dp0.1_0.5_0.5_init_lr0.01 \
# --move_out_path /files/yxue/research/allstate/out/omninet/struct_domain_img_text_encoding_as_struct/vqa_struct_norm_bs64_struct_periph_3_dp0.1_0.5_0.5_init_lr0.01 \
# --structured_folder structured_cache_type_temp_cos_sim_0_multi_cache_topN_data_seed951 \
# --inject_at_encoder --inject_at_logits \
# --n_workers 2 --init_lr 0.01 --struct_periph_dropout 0.1 --struct_temp_periph_dropout 0.5 \
# --struct_spat_periph_dropout 0.5 < /dev/null > /dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python train.py 841110 vqa_struct 64 --n_gpus 1 \
# --save_interval 6470 --eval_interval 6470 --conf_type vqa_struct \
# --model_save_path /files/yxue/research/allstate/out/omninet/struct_domain_img_text_encoding_as_struct/vqa_struct_norm_bs64_struct_periph_3_dp0.1_0.5_0.5_init_lr0.01_spat \
# --move_out_path /files/yxue/research/allstate/out/omninet/struct_domain_img_text_encoding_as_struct/vqa_struct_norm_bs64_struct_periph_3_dp0.1_0.5_0.5_init_lr0.01_spat \
# --structured_folder structured_cache_type_spat_cos_sim_0_multi_cache_topN_data_seed951 \
# --inject_at_encoder --inject_at_logits \
# --n_workers 2 --init_lr 0.01 --struct_periph_dropout 0.1 --struct_temp_periph_dropout 0.5 \
# --struct_spat_periph_dropout 0.5 < /dev/null > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python train.py 841110 vqa_struct 64 --n_gpus 1 \
--save_interval 6470 --eval_interval 6470 --conf_type vqa_struct \
--model_save_path /files/yxue/research/allstate/out/omninet/struct_domain_img_text_encoding_as_struct/vqa_struct_norm_bs64_temp_only_at_logits_struct_periph_3_init_lr0.01 \
--move_out_path /files/yxue/research/allstate/out/omninet/struct_domain_img_text_encoding_as_struct/vqa_struct_norm_bs64_temp_only_at_logits_struct_periph_3_init_lr0.01 \
--structured_folder structured_cache_type_temp_cos_sim_0_multi_cache_topN_data_seed951 \
--inject_at_logits \
--n_workers 2 --init_lr 0.01 \
--save_best --unstructured_as_structured < /dev/null > /dev/null 2> /struct_domain_img_text_encoding_as_struct_vqa_struct_norm_bs64_temp_only_at_logits_struct_periph_3_init_lr0.01.err &

CUDA_VISIBLE_DEVICES=1 nohup python train.py 841110 vqa_struct 64 --n_gpus 1 \
--save_interval 6470 --eval_interval 6470 --conf_type vqa_struct \
--model_save_path /files/yxue/research/allstate/out/omninet/struct_domain_img_text_encoding_as_struct/vqa_struct_norm_bs64_temp_struct_periph_3_init_lr0.02 \
--move_out_path /files/yxue/research/allstate/out/omninet/struct_domain_img_text_encoding_as_struct/vqa_struct_norm_bs64_temp_struct_periph_3_init_lr0.02 \
--structured_folder structured_cache_type_temp_cos_sim_0_multi_cache_topN_data_seed951 \
--inject_at_encoder --inject_at_logits \
--n_workers 2 --init_lr 0.02 \
--save_best --unstructured_as_structured < /dev/null > /dev/null 2> /struct_domain_img_text_encoding_as_struct_vqa_struct_norm_bs64_temp_struct_periph_3_init_lr0.02.err &
