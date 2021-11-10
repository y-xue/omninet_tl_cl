#!/bin/bash

cdn=/files/yxue/research/allstate
test(){
	model_name=$1 
	
	CUDA_VISIBLE_DEVICES=0 python evaluate.py vqa_struct \
	${cdn}/out/omninet/${model_name}/model.pth \
	--batch_size 64 --structured_path ${cdn}/data/vqa/synthetic_structured_clustering_std3 \
	--out_fn ${cdn}/out/omninet/${model_name}_predictions.json \
	--inject_at_logits --fusion > ${cdn}/out/omninet/${model_name}_predictions.log;
}

test1(){
	model_name=$1
	fusion_attn_type=$2
	
	CUDA_VISIBLE_DEVICES=0 python evaluate.py vqa_struct \
	${cdn}/out/omninet/${model_name}/model.pth \
	--batch_size 64 --structured_path ${cdn}/data/vqa/synthetic_structured_clustering_std3 \
	--out_fn ${cdn}/out/omninet/${model_name}_predictions.json \
	--inject_at_logits --inject_at_encoder \
	--fusion_attn_type ${fusion_attn_type} --fusion > ${cdn}/out/omninet/${model_name}_predictions.log;
}

test2(){
	model_name=$1
	fusion_attn_type=$2
	
	CUDA_VISIBLE_DEVICES=0 python evaluate.py vqa_struct \
	${cdn}/out/omninet/${model_name}/model.pth \
	--batch_size 64 --structured_path ${cdn}/data/vqa/synthetic_structured_clustering_std3 \
	--out_fn ${cdn}/out/omninet/${model_name}_predictions.json \
	--inject_at_logits --inject_after_encoder \
	--fusion_attn_type ${fusion_attn_type} --fusion > ${cdn}/out/omninet/${model_name}_predictions.log;
}

test3(){
	model_name=$1
	fusion_attn_type=$2
	
	CUDA_VISIBLE_DEVICES=0 python evaluate.py vqa_struct \
	${cdn}/out/omninet/${model_name}/model.pth \
	--batch_size 64 --structured_path ${cdn}/data/vqa/synthetic_structured_clustering_std3 \
	--out_fn ${cdn}/out/omninet/${model_name}_predictions.json \
	--inject_at_logits --inject_after_encoder \
	--fusion_attn_type ${fusion_attn_type} \
	--convex_gate --fusion > ${cdn}/out/omninet/${model_name}_predictions.log;
}

test4(){
	model_name=$1
	spat_fusion_attn_type=$2
	temp_fusion_attn_type=$3
	
	CUDA_VISIBLE_DEVICES=3 python evaluate.py vqa_struct \
	${cdn}/out/omninet/${model_name}/model.pth \
	--batch_size 64 --structured_path ${cdn}/data/vqa/synthetic_structured_clustering_std3 \
	--out_fn ${cdn}/out/omninet/${model_name}_predictions.json \
	--inject_at_logits --inject_at_encoder \
	--spat_fusion_attn_type ${spat_fusion_attn_type} \
	--temp_fusion_attn_type ${temp_fusion_attn_type} \
	--convex_gate --fusion > ${cdn}/out/omninet/${model_name}_predictions.log;
}

test_struct_domain(){
	model_name=$1
	dp=$2

	CUDA_VISIBLE_DEVICES=3 python evaluate.py vqa_struct \
	${cdn}/out/omninet/${model_name}/model.pth \
	--batch_size 64 --structured_path ${cdn}/data/vqa/synthetic_structured_clustering_std3 \
	--out_fn ${cdn}/out/omninet/${model_name}_predictions.json \
	--inject_at_logits \
	--struct_periph_dropout ${dp} > ${cdn}/out/omninet/${model_name}_predictions.log;
}

# test vqa_struct_norm_bs64_clustering_std3_inject_struct_at_logits_556334;
# sleep 10;
# test1 vqa_struct_norm_bs64_encoder_fusion_gating_per_frame_582210 selective;
# sleep 10;
# test1 vqa_struct_norm_bs64_encoder_gated_attention_fusion_252329 gated;

# test1 vqa_struct_norm_bs64_encoder_gated_attention_fusion_with_dropout_258799 gated;
# sleep 10;
# test2 vqa_struct_norm_bs64_inject_after_encoder_gated_attention_fusion_with_dropout_252329 gated;
# sleep 10;
# test1 vqa_struct_norm_bs64_encoder_gated_attention_fusion_no_gating_317029 none;
# sleep 10;
# test3 vqa_struct_norm_bs64_inject_after_encoder_selective_attention_fusion_575829 selective;
# test4 vqa_struct_norm_bs64_encoder_fusion_gating_per_frame_softmax_convex_634059 selective selective;

test_struct_domain struct_domain/vqa_struct_norm_bs64_dp0.8_init_lr0.01_414079 0.8
