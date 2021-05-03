import os
import pickle
from libs.utils import dataloaders as dl

mm_image_dir = '/mnt-gluster/data/yxue/research/allstate/data/cifar2'
mm_text_dir = '/mnt-gluster/data/yxue/research/allstate/data/aclImdb_new'
mm_video_dir = '/mnt-gluster/data/yxue/research/allstate/data/hmdb2'
mm_video_process_dir = '/mnt-gluster/data/yxue/research/allstate/data/hmdb2process_val'
mm_dir = '/mnt-gluster/data/yxue/research/allstate/data/synthetic_mm_cifar_imdb_hmdb'

batch_size=5

seq_lst = ['0-1-0', '0-2-0', '1-0-0', '1-1-0', '1-2-0', '2-1-0', 
                    '0-1-1', '1-2-1', '2-2-0', '2-2-1', '0-2-1', '1-1-1', 
                    '0-0-1', '2-0-0', '2-1-1', '1-0-1']
full_seq = 'ITTIV'
with open(os.path.join(mm_dir, 'sample_idx_ITTIV_pT.9_35k-15k-15k'), 'rb') as f:
    sample_idx = pickle.load(f)

dl_lst, val_dl_lst, test_dl_lst = dl.mm_batchgen(full_seq, seq_lst, mm_image_dir, mm_text_dir, mm_video_dir, mm_video_process_dir, sample_idx, num_workers=0, batch_size=batch_size)

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

DLS = iter(cycle(dl_lst))
seq_it = iter(cycle(seq_lst))

def next_batch():
	seq = next(seq_it)
	print(seq)
	DL = next(DLS)
	batch = next(DL)
	return batch


from libs.omninet.peripherals import *
image_input_perph_pretrained_path = '/mnt-gluster/data/yxue/research/allstate/out/cifar2_res/cifar2_mbnt_mydl_0.01lr_decay_128bs300epochs.pt'
video_input_perph_pretrained_path = '/mnt-gluster/data/yxue/research/allstate/out/hmdb2_res/hmdb2_convlstm_32hidden_channel_56max_pool_8bs50epochs.pt'
english_language_perph_pretrained_path = '/mnt-gluster/data/yxue/research/allstate/out/transformer_clf_bertEmb/transformer_clf_bertEmb_mydl_2layer2head64d128dinner0.1dropout_0.0001lr64bs20epochs.pt'


# 443M
image_input_perph = ImageInputPeripheralMobilenetV2(output_dim=64,dropout=0.1,pretrained_path=image_input_perph_pretrained_path,freeze_layers=True)

# 526M
english_language_perph = LanguagePeripheralTSFM(output_dim=64,embed_dim=64,dropout=0.1,pretrained_path=english_language_perph_pretrained_path)

# 435M
video_input_perph = VideoInputPeripheralConvLSTM(output_dim=64,dropout=0.1,pretrained_path=video_input_perph_pretrained_path,freeze_layers=True)

image_input_perph = image_input_perph.cuda(0)
imgs = batch['imgs'].cuda(0)
image_encodings = image_input_perph.encode(imgs[:,0,:,:,:].squeeze(1))
# image_encodings: [5, 1, 16, 64]) [b,t,s,f]

english_language_perph=english_language_perph.cuda(0)
texts = [t[0] for t in batch['text']]
embs, pad_mask = english_language_perph.embed_sentences(texts, tokenized=False)
# embs: [5, 512, 1, 64] [b,t,s,f]
# pad_mask: [5, 512]

# 328M for 5 videos
video_input_perph=video_input_perph.cuda(0)
videos=batch['videos'].cuda(0)
video_enc = video_input_perph.encode(videos[:,0,:,:,:,:].squeeze(1))
# video_enc: [5, 16, 16, 64] [b,t,s,f]
