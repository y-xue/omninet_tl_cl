from libs.utils import dataloaders as dl
import os
import json
import pickle

data_path='/files/yxue/research/allstate/data'

mm_image_dir = os.path.join(data_path, 'cifar2')
mm_text_dir = os.path.join(data_path, 'aclImdb_new')
mm_video_dir = os.path.join(data_path, 'hmdb2')
mm_video_process_dir = os.path.join(data_path, 'hmdb2process_val')

mm_dir = os.path.join(data_path, 'synthetic_mm_cifar_imdb_hmdb')

sample_idx_fn='sample_idx_ITTIV_pT.7_no_dup_img_text_video_seed10'
with open(os.path.join(mm_dir, sample_idx_fn), 'rb') as f:
	sample_idx = pickle.load(f)


seq_lst = list(sample_idx['test'].keys())
full_seq = 'ITTIV'
sample_weights_fn='sample_weights_ITTIV_0.json'

with open(os.path.join(mm_dir, sample_weights_fn), 'r') as f:
	predefined_sample_weights = json.load(f)

n_workers=0
batch_size=5
norm_weights=False
scale_weights=False
pick_sample_by_intsance_id=False
stop_overfitted_ins=False

dl_lst, val_dl_lst, test_dl_lst = dl.mm_batchgen(full_seq, seq_lst, 
    mm_image_dir, mm_text_dir, mm_video_dir, mm_video_process_dir,
    predefined_sample_weights, sample_idx, 
    num_workers=n_workers, batch_size=batch_size, 
    norm_weights=norm_weights, scale_weights=scale_weights,
    pick_sample_by_intsance_id=pick_sample_by_intsance_id,
    stop_overfitted_ins=stop_overfitted_ins)

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

DLS = [iter(cycle(tr_dl)) for tr_dl in dl_lst]

print(seq_lst)

DL=DLS[3]
batch = next(DL)

# print(batch['text'])
print(batch['imgs'][3])