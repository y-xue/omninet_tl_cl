from pick_caches import *

# structured_fn = '/mnt-gluster/data/yxue/research/allstate/out/synthetic_structured_data_iter38360.dict'
# out_path = '/mnt-gluster/data/yxue/research/allstate/out'

# structured_fn = '/mnt-gluster/data/yxue/research/allstate/out/omninet/vqa_select_cache_invocab_10epochs/synthetic_structured_data_iter100900.dict'
# out_path = '/mnt-gluster/data/yxue/research/allstate/out/omninet/vqa_select_cache_invocab_10epochs_output/'
# pick_caches(structured_fn, out_path)

# structured_fn = '/mnt-gluster/data/yxue/research/allstate/out/omninet/vqa_select_cache_from_self_trained_vqa_bs64_at_78epoch_cache_dicts/synthetic_structured_data_iter129329.dict'
# out_path = '/mnt-gluster/data/yxue/research/allstate/out/omninet/vqa_select_cache_from_self_trained_vqa_bs64_at_78epoch_cache_dicts/'
# pick_caches(structured_fn, out_path, random_cache_type=False, cos_sim=False, multi_cache='topN')

# cp /mnt-gluster/data/yxue/research/allstate/out/omninet/vqa_select_cache_from_self_trained_vqa_bs64_at_78epoch_cache_dicts/synthetic_train_structured_random_cache_type_0_cos_sim_0_multi_cache_topN_data_seed951.dict /files/yxue/research/allstate/data/vqa/structured_random_cache_type_0_cos_sim_0_multi_cache_topN_data_seed951/synthetic_train_structured_data.dict
# cp /mnt-gluster/data/yxue/research/allstate/out/omninet/vqa_select_cache_from_self_trained_vqa_bs64_at_78epoch_cache_dicts/synthetic_val_structured_random_cache_type_0_cos_sim_0_multi_cache_topN_data_seed951.dict /files/yxue/research/allstate/data/vqa/structured_random_cache_type_0_cos_sim_0_multi_cache_topN_data_seed951/synthetic_val_structured_data.dict


structured_fn = '/files/yxue/research/allstate/out/omninet/vqa_select_cache_from_self_trained_vqa_bs64_at_78epoch_cache_dicts/synthetic_structured_data_iter129329.dict'
out_path = '/files/yxue/research/allstate/out/omninet/vqa_select_cache_from_self_trained_vqa_bs64_at_78epoch_cache_dicts'
pick_caches(structured_fn, out_path, cache_type='temp', cos_sim=False, multi_cache='topN')

pick_caches(structured_fn, out_path, cache_type='spat', cos_sim=False, multi_cache='topN')
