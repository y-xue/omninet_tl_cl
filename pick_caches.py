import pickle
import numpy as np
import json
import os
from tqdm import tqdm

# vqa_train_ques='/data/vqa/v2_OpenEnded_mscoco_train2014_questions.json'
# vqa_train_ques='/mnt-gluster/data/yxue/research/omninet_data/vqa/v2_OpenEnded_mscoco_train2014_questions.json'
vqa_train_ques='/files/yxue/research/allstate/data/vqa/v2_OpenEnded_mscoco_train2014_questions.json'

COS_SIM = 0.9
SEED = 951

np.random.seed(SEED)

def pick_caches(structured_fn, out_path, cache_type='random', cos_sim=True, multi_cache='top1'):
	with open(structured_fn, 'rb') as f:
		d = pickle.load(f)

	with open(vqa_train_ques, 'r') as f:
		ques = json.load(f)

	num_tr = len(ques['questions'])

	tr_structured = {}
	val_structured = {}

	for k in tqdm(d[multi_cache]['temporal']):
		if multi_cache == 'top1':
			spatial_structured = d[multi_cache]['spatial'][k][0]
			temporal_structured = d[multi_cache]['temporal'][k][0]
		else:
			spatial_structured = np.mean(d[multi_cache]['spatial'][k], 0)
			temporal_structured = np.mean(d[multi_cache]['temporal'][k], 0)

		if cache_type == 'random':
			if np.random.random() < 0.5:
				structured = spatial_structured
			else:
				structured = temporal_structured
		elif cache_type == 'all':
			structured = np.vstack([temporal_structured, spatial_structured])
		elif cache_type == 'temp':
			structured = temporal_structured
		elif cache_type == 'spat':
			structured = spatial_structured

		if cos_sim:
			if random_cache_type:
				structured = rand_cos_sim(structured, COS_SIM)
			else:
				structured[0] = rand_cos_sim(structured[0], COS_SIM)
				structured[1] = rand_cos_sim(structured[1], COS_SIM)

		if k < num_tr:
			tr_structured[k] = structured
		else:
			val_structured[k - num_tr] = structured

	with open(os.path.join(out_path, 'synthetic_train_structured_cache_type_%s_cos_sim_%d_multi_cache_%s_data_seed%s.dict'%(cache_type,cos_sim,multi_cache,SEED)), 'wb') as f:
		pickle.dump(tr_structured, f)

	with open(os.path.join(out_path, 'synthetic_val_structured_cache_type_%s_cos_sim_%d_multi_cache_%s_data_seed%s.dict'%(cache_type,cos_sim,multi_cache,SEED)), 'wb') as f:
		pickle.dump(val_structured, f)


	# with open(os.path.join(out_path, 'synthetic_train_structured_random_cache_type_%d_cos_sim_%d_multi_cache_%s_data_seed%s.dict'%(random_cache_type,cos_sim,multi_cache,SEED)), 'wb') as f:
	# 	pickle.dump(tr_structured, f)

	# with open(os.path.join(out_path, 'synthetic_val_structured_random_cache_type_%d_cos_sim_%d_multi_cache_%s_data_seed%s.dict'%(random_cache_type,cos_sim,multi_cache,SEED)), 'wb') as f:
	# 	pickle.dump(val_structured, f)

def rand_cos_sim(v, costheta):
    # Form the unit vector parallel to v:
    u = v / np.linalg.norm(v)
    # Pick a random vector:
    r = np.random.multivariate_normal(np.zeros_like(v), np.eye(len(v)))
    # Form a vector perpendicular to v:
    uperp = r - r.dot(u)*u
    # Make it a unit vector:
    uperp = uperp / np.linalg.norm(uperp)
    # w is the linear combination of u and uperp with coefficients costheta
    # and sin(theta) = sqrt(1 - costheta**2), respectively:
    w = costheta*u + np.sqrt(1 - costheta**2)*uperp
    return w
