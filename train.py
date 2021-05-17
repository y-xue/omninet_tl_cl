#
# Copyright 2019 Subhojeet Pramanik, Aman Husain, Priyanka Agrawal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ======================================================================
"""
Authors: Subhojeet Pramanik

OmniNet training script. 

"""
import argparse
import os
import torch
import time
import glob
import numpy as np
import libs.omninet as omninet
from libs.utils import dataloaders as dl
from tensorboardX import SummaryWriter
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import libs.omninet.routines as r
from libs.omninet.util import ScheduledOptim
from torch.optim.adam import Adam
from torch.optim import SGD
import random
import sys
import pickle
from tqdm import tqdm
from libs.utils.train_util import *
import json
import shutil
import re

from omninet_config import *
from libs.utils.reward_func import *

parser = argparse.ArgumentParser(description='OmniNet training script.')
parser.add_argument('n_iters', help='Number of iterations to train.')
parser.add_argument('tasks', help='List of tasks seperated by comma.')
parser.add_argument('batch_sizes', help='List of batch size for each task seperated by comma')
parser.add_argument('--n_jobs', default=1, help='Number of asynchronous jobs to run for each task.')
parser.add_argument('--n_gpus', default=1, help='Number of GPUs to use')
parser.add_argument('--n_workers', default=0, type=int, help='Number of workers to load data')
parser.add_argument('--save_interval', default=100, help='Number of iterations after which to save the model.')
parser.add_argument('--restore', default=-1, help='Step from which to restore model training')
parser.add_argument('--restore_last', help='Restore the latest version of the model.', action='store_true')
parser.add_argument('--eval_interval', help='Interval after which to evaluate on the test/val set.', default=1000)
parser.add_argument('--model_save_path', default='/out/test', help='path to save the model.')
parser.add_argument('--move_out_path', default=None, help='path to save the best model.')
parser.add_argument('--with_val', action='store_true', help='True if run on train/val/test split.')
parser.add_argument('--structured_folder', default='synthetic_structured_clustering_std2_valAcc66', help='path to structured data.')
parser.add_argument('--norm_weights', action='store_true', help='True if normalize sample weights by instance.')
parser.add_argument('--scale_weights', action='store_true', help='True if scale weights by total number of modalities.')
parser.add_argument('--sample_weights_fn', default='sample_weights_1.json', help='file name of predefined sample weights.')
parser.add_argument('--sample_idx_fn', default='sample_idx_ITTIV_pT.9_35k-15k-15k', help='file name of sample_idx.')
parser.add_argument('--peripherals_type', default='default', help='Choose peripherals types')
parser.add_argument('--conf_type', default='default', help='Choose confurigation types')
parser.add_argument('--torch_seed', default=47, type=int, help='torch manual_seed')
parser.add_argument('--random_seed', default=983, type=int, help='random seed')
parser.add_argument('--numpy_seed', default=1024, type=int, help='numpy seed')
parser.add_argument('--data_seed', default=615, type=int, help='numpy seed')
parser.add_argument('--all_seed', default=1029, type=int, help='seed')
parser.add_argument('--save_best', action='store_true', help='True if only save the model on validation.')
parser.add_argument('--rewarding', action='store_true', help='True if evaluating model with reward function.')
parser.add_argument('--skip_seqs', default=None, nargs='+', help='skill seq in training')
parser.add_argument('--overfitting_threshold', default=1, type=float, help='threshold to determine if model is overfitting')
parser.add_argument('--overfitting_start', default=1000, type=float, help='iterations when to start handling overfitting')
parser.add_argument('--stop_overfitted_ins', action='store_true', help='True if stop training on instances that contains samples of overfitted sequence types.')
parser.add_argument('--detect_overfitting_with_tr_acc', action='store_true', help='True if determine overfitting by looking at training accuracy.')
#parser.add_argument('--eval_first', action='store_true', help='True if evaluate model in the first iteration.')
parser.add_argument('--data_path', default='/files/yxue/research/allstate/data', help='data path')
parser.add_argument('--random_seq', action='store_true', help='Deprecated. True if randomly pick a seq to train in each iteration.')
parser.add_argument('--testing', action='store_true', help='True if training with validation set for a few epochs for testing.')
parser.add_argument('--notest', action='store_true', help='True if no test model.')
parser.add_argument('--test', action='store_true', help='True if only test model.')
parser.add_argument('--evaluate', action='store_true', help='True if only eval model.')
parser.add_argument('--pick_sample_by_intsance_id', action='store_true', help='True if pick samples groups of data; grouping by instance id.')
parser.add_argument('--pick_seq', default='prob', help='choice of how to pick a seq to train in next iteration. '
    'prob: picking seq by probablities proportional to sample sizes. random: picking randomly.'
    'sequential: picking seq following a fixed order')
parser.add_argument('--inject_at_logits', action='store_true', help='True if inject structured feature at final logit layer.')
parser.add_argument('--inject_at_encoder', action='store_true', help='True if inject structured feature at encoder.')
parser.add_argument('--inject_after_encoder', action='store_true', help='True if inject structured feature right after encoder.')
parser.add_argument('--inject_at_decoder', action='store_true', help='True if inject structured feature at decoder.')
parser.add_argument('--temp_fusion_attn_type', default='none', type=str, help='Temporal attention type in fusion.')
parser.add_argument('--spat_fusion_attn_type', default='none', type=str, help='Spatial attention type in fusion.')
parser.add_argument('--convex_gate', action='store_true', help='True if use convex gate.')
parser.add_argument('--use_sgd', action='store_true', help='True if use SGD.')
parser.add_argument('--prob_fn', default=None, type=str, help='file name of predefined sampling probablities by sequences')
parser.add_argument('--init_lr', default=0.02, type=float, help='init_lr')
parser.add_argument('--use_boxes', action='store_true', help='True if utilize bounding boxes.')
parser.add_argument('--pooling', action='store_true', help='True if use the whole image feature vector for gating.')
parser.add_argument('--fusion', action='store_true', help='True if fuse structured features with unstructured features by attention-based fusion.')
parser.add_argument('--struct_periph_dropout', default=0.1, type=float, help='dropout rates at struct peripherals')
parser.add_argument('--struct_temp_periph_dropout', default=0.1, type=float, help='dropout rates at struct temp peripherals')
parser.add_argument('--struct_spat_periph_dropout', default=0.1, type=float, help='dropout rates at struct spat peripherals')
parser.add_argument('--no_logit_struct_peripheral', action='store_true', help='True if no struct peripheral at logits.')
parser.add_argument('--unstructured_as_structured', action='store_true', help='Indicates if using unstructured encodings as structured data.')
parser.add_argument('--unfreeze', default=[], nargs='+', type=str, help='indicates which peripheral to unfreeze')
parser.add_argument('--logit_struct_periph_dim', default=512, type=int, help='logit_struct_periph_dim')
parser.add_argument('--n_warmup_steps', default=16000, type=int, help='n_warmup_steps for ScheduledOptim')

args = parser.parse_args()

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(int(args.all_seed))

# data_path = '/files/yxue/research/allstate/data'
# data_path = '/mnt-gluster/data/yxue/research/allstate/data'

data_path = args.data_path #'/files/yxue/research/allstate/data'

coco_images = os.path.join(data_path, 'coco/train_val')
caption_dir = os.path.join(data_path, 'coco')
vqa_dir = os.path.join(data_path, 'vqa')
# structured_path = os.path.join(data_path, 'vqa/structured_seed951')
# structured_path = os.path.join(data_path, 'vqa/structured_random_cache_type_0_cos_sim_0_multi_cache_topN_data_seed951')
structured_path = os.path.join(data_path, 'vqa', args.structured_folder)
# hmdb_data_dir = os.path.join(data_path, 'hmdb')
hmdb_data_dir = data_path
hmdb_process_dir = os.path.join(data_path, 'hmdbprocess')
# penn_data_dir = os.path.join(data_path, 'penn')

# coco_images = '/data/coco/train_val'
# caption_dir = '/data/coco'
# vqa_dir = '/data/vqa'
# # model_save_path = 'checkpoints'
# hmdb_data_dir='/data/hmdb'
# hmdb_process_dir='/data/hmdbprocess'
# penn_data_dir='/data/penn'
# structured_path = '/data/vqa/structured_seed951'


# coco_images = '/data/gluster/brick1/yxue/research/omninet/data/coco/train_val'
# caption_dir = '/data/gluster/brick1/yxue/research/omninet/data/coco'
# vqa_dir = '/data/gluster/brick1/yxue/research/omninet/data/vqa'
# model_save_path = '/data/gluster/brick1/yxue/research/omninet/checkpoints/mm_vqa_labelFirst_test'
# hmdb_data_dir='/data/gluster/brick1/yxue/research/omninet/data/hmdb'
# hmdb_process_dir='/data/gluster/brick1/yxue/research/omninet/data/hmdbprocess'
# #hmdb_process_dir=''
# penn_data_dir='/data/gluster/brick1/yxue/research/omninet/data/penn'

# coco_images = '/mnt-gluster/data/yxue/research/omninet_data/coco/train_val'
# caption_dir = '/mnt-gluster/data/yxue/research/omninet_data/coco'
# vqa_dir = '/mnt-gluster/data/yxue/research/omninet_data/vqa'
# hmdb_data_dir='/mnt-gluster/data/yxue/research/omninet_data/hmdb'
# hmdb_process_dir='/mnt-gluster/data/yxue/research/omninet_data/hmdbprocess'
# #hmdb_process_dir=''
# penn_data_dir='/mnt-gluster/data/yxue/research/omninet_data/penn'

# coco_images = '/data/gluster/brick1/yxue/research/omninet/data/coco/train_val'
# caption_dir = '/data/gluster/brick1/yxue/research/omninet/data/coco'
# vqa_dir = '/data/gluster/brick1/yxue/research/omninet/data/vqa'
# hmdb_data_dir='/data/gluster/brick1/yxue/research/omninet/data/hmdb'
# hmdb_process_dir='/data/gluster/brick1/yxue/research/omninet/data/hmdbprocess'
# #hmdb_process_dir=''
# penn_data_dir='/data/gluster/brick1/yxue/research/omninet/data/penn'
# birds_dir='/home/yxue/research/allstate/data/CUB_200_2011'
# birds_dir='/mnt-gluster/data/yxue/research/allstate/data/CUB_200_2011'
birds_dir='/CUB_200_2011'

# data_path = '/mnt-gluster/data/yxue/research/allstate/data'


# mm_image_dir = os.path.join(data_path, 'cifar2')
# mm_text_dir = os.path.join(data_path, 'aclImdb_new')
# mm_video_dir = os.path.join(data_path, 'hmdb2')
# mm_video_process_dir = os.path.join(data_path, 'hmdb2process_val')
mm_image_dir = data_path
mm_text_dir = os.path.join(data_path, 'aclImdb_new')
mm_video_dir = data_path
mm_video_process_dir = os.path.join(data_path, 'hmdbprocess')

# mm_video_dir = '/mnt-gluster/data/yxue/research/omninet_data/hmdb2'
# mm_video_process_dir = '/mnt-gluster/data/yxue/research/omninet_data/hmdb2process'
# mm_video_dir = '/mnt-gluster/data/yxue/research/allstate/data/hmdb2'
# mm_video_process_dir = '/mnt-gluster/data/yxue/research/allstate/data/hmdb2process'
mm_dir = os.path.join(data_path, 'synthetic_mm_cifar_imdb_hmdb')

# sample_idx_fn = 'sample_idx_ITTITITTTV'
# sample_idx_fn = 'sample_idx_ITTIV_pT.9_35k-15k-15k'
# sample_idx_fn = 'sample_idx_ITTIV_pT.9_35k-15k-15k_seed1024'
# sample_idx_fn = 'sample_idx_ITTIV_pT.7_30k-10k-10k_adjusted_sample_sizes_seed10'

# # model_save_path = '/home/yxue/research/allstate/code/omninet_struct/checkpoints/birds_struct' #'/mnt-gluster/data/yxue/research/allstate/omninet_struct/checkpoints/birds'
# model_save_path = '/mnt-gluster/data/yxue/research/allstate/omninet_struct/checkpoints/mm_ITIT_p.6.6_100000' #mm_vqa_labelFirst_seqMiniBatch_test'

def send_to_device(x, gpu_id):
    if x is not None:
        return x.cuda(device=gpu_id)
    return x

def print_log(s, fn):
    with open(fn, 'a') as f:
        print(s, file=f)

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def evaluating(log, eval_interval, i):
    if not log or eval_interval is None or i % eval_interval != 0:
        return False

    return True

def train(shared_model, task, batch_size, train_steps, gpu_id, start,  restore,  counter, barrier=None, 
          n_workers=0, save_interval=None,
          eval_interval=None, eval_first=False, log=True, peripherals_type='default', conf_type='default', 
          model_save_path='/out', move_out_path='/mnt-gluster/data/yxue/research/allstate/out', 
          save_best=False, testing=False, with_val=False, norm_weights=False, scale_weights=False, 
          sample_weights_fn=None, sample_idx_fn=None, rewarding=False, skip_seqs=None,
          overfitting_threshold=1, overfitting_start=1000, stop_overfitted_ins=False,
          detect_overfitting_with_tr_acc=False,
          random_seq=False, notest=False, test=False, evaluate=False,
          pick_sample_by_intsance_id=False, pick_seq='prob', use_sgd=False, prob_fn=None):
    log_dir = 'logs/%s' % task
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if (log == True):
        summary_writer = SummaryWriter(log_dir)
    # Create local model
     
    # torch.manual_seed(int(random.random() * 1000))
    if gpu_id>0:
        if conf_type == 'default':
            config = defaultconf()
            # model = omninet.OmniNet(gpu_id=gpu_id, peripherals_type=peripherals_type)
        elif conf_type == 'timeline':
            print('using timeline_conf')
            # model = omninet.OmniNet(gpu_id=gpu_id, config=timeline_conf() , peripherals_type=peripherals_type)
            config = timeline_conf()
        elif conf_type == 'vqa_struct':
            print('using vqa_struct_config')
            config = vqa_struct_config()

            config[0]['temp_fusion_attn_type'] = args.temp_fusion_attn_type
            config[0]['spat_fusion_attn_type'] = args.spat_fusion_attn_type
            config[0]['inject_at_logits'] = args.inject_at_logits
            config[0]['inject_at_encoder'] = args.inject_at_encoder
            config[0]['inject_after_encoder'] = args.inject_after_encoder
            config[0]['convex_gate'] = args.convex_gate
            config[0]['pooling'] = args.pooling
            config[0]['no_logit_struct_peripheral'] = args.no_logit_struct_peripheral
            config[0]['logit_struct_periph_dim'] = args.logit_struct_periph_dim
            config[1]['struct_dropout'] = args.struct_periph_dropout
            config[1]['struct_temp_dropout'] = args.struct_temp_periph_dropout
            config[1]['struct_spat_dropout'] = args.struct_spat_periph_dropout
            if args.inject_at_decoder:
                config[0]['inject_at_decoder'] = True
                config[0]['use_s_decoder'] = True

            for periph_name in args.unfreeze:
                config[1]['unfreeze'][periph_name] = True
        
        if restore == 0:
            print_log(config, move_out_path + '.log')
        # print(config)
        model = omninet.OmniNet(gpu_id=gpu_id, config=config , peripherals_type=peripherals_type, unstructured_as_structured=args.unstructured_as_structured)
        model=model.cuda(gpu_id)
    else:
        #For GPU 0, use the shared model always
        model=shared_model

    if task == 'caption':
        DL,val_dl = dl.coco_cap_batchgen(caption_dir=caption_dir, image_dir=coco_images,
                                  num_workers=n_workers,
                                  batch_size=batch_size)
        
        optimizer = ScheduledOptim(
            Adam(
                filter(lambda x: x.requires_grad, shared_model.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            512, args.n_warmup_steps,restore,init_lr=args.init_lr)
    elif task == 'vqa' or task == 'vqa_struct':
        if task == 'vqa':
            tr_dl,val_dl,test_dl = dl.vqa_batchgen(vqa_dir, coco_images, num_workers=n_workers, batch_size=batch_size, with_val=with_val, use_boxes=args.use_boxes, make_tr_dl_iter=False, unstructured_as_structured=args.unstructured_as_structured, data_seed=args.data_seed)
        else:
            tr_dl,val_dl,test_dl = dl.vqa_batchgen(vqa_dir, coco_images, num_workers=n_workers, batch_size=batch_size, structured_path=structured_path, with_val=with_val, use_boxes=args.use_boxes, make_tr_dl_iter=False, unstructured_as_structured=args.unstructured_as_structured, data_seed=args.data_seed)
            spat_struct_enc_gate_tr = {}
            temp_struct_enc_gate_tr = {}
            spat_struct_enc_gate_val = {}
            temp_struct_enc_gate_val = {}
            spat_struct_enc_gate_test = {}
            temp_struct_enc_gate_test = {}
            emb_loss_spat = None
            emb_loss_temp = None

        DL = iter(cycle(tr_dl))

        if use_sgd:
            print('SGD')
            optimizer = ScheduledOptim(
                SGD(
                    filter(lambda x: x.requires_grad, shared_model.parameters()),
                    lr=0.0001, momentum=0.9),
                512, args.n_warmup_steps,restore,max_lr=0.0001,init_lr=args.init_lr)
        else:
            optimizer = ScheduledOptim(
                Adam(
                    filter(lambda x: x.requires_grad, shared_model.parameters()),
                    betas=(0.9, 0.98), eps=1e-09),
                512, args.n_warmup_steps,restore,max_lr=0.0001,init_lr=args.init_lr)

    # elif task == 'vqa_struct':
    #     DL,val_dl = dl.vqa_struct_batchgen(vqa_dir, coco_images, num_workers=0, batch_size=batch_size)
    #     optimizer = ScheduledOptim(
    #         Adam(
    #             filter(lambda x: x.requires_grad, shared_model.parameters()),
    #             betas=(0.9, 0.98), eps=1e-09),
    #         512, 16000,restore,max_lr=0.0001,init_lr=args.init_lr)
    elif task == 'birds' or task == 'birds_struct':
        DL,val_dl,test_dl = dl.birds_batchgen(birds_dir, num_workers=n_workers, batch_size=batch_size, testing=testing)
        optimizer = ScheduledOptim(
            Adam(
                filter(lambda x: x.requires_grad, shared_model.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            512, args.n_warmup_steps,restore,max_lr=0.0001,init_lr=args.init_lr)
    elif task == 'mm_ITV' or task == 'mm_seq10':
        with open(os.path.join(mm_dir, sample_idx_fn), 'rb') as f:
            sample_idx = pickle.load(f)

        if task == 'mm_ITV':
            seq_lst = list(sample_idx['test'].keys())
            # seq_lst = ['1-0-0', '0-1-0', '0-0-1',
            #            '1-1-0', '0-2-0', '2-0-0', '1-0-1', '0-1-1',
            #            '1-2-0', '2-1-0', '1-1-1', '0-2-1', '2-0-1',
            #            '2-2-0', '1-2-1', '2-1-1', 
            #            '2-2-1']
            # if skip_seq:
            #     # # old dataset doesn't have data of seq 2-0-1
            #     # seq_lst = ['1-0-0', '0-1-0', '0-0-1',
            #     #        '1-1-0', '0-2-0', '2-0-0', '1-0-1', '0-1-1',
            #     #        '1-2-0', '2-1-0', '1-1-1', '0-2-1',
            #     #        '2-2-0', '1-2-1', '2-1-1', 
            #     #        '2-2-1']
            full_seq = 'ITTIV'
        elif task == 'mm_seq10':
            seq_lst = ['0-1-0', '0-2-0', '1-2-0', '1-3-0', '2-3-0', '2-4-0', 
                    '2-5-0', '2-6-0', '0-3-0', '0-4-0', '0-5-0', '0-6-0', 
                    '1-0-0', '1-1-0', '1-4-0', '1-5-0', '1-6-0', '1-4-1', 
                    '1-6-1', '0-6-1', '2-2-0', '0-5-1', '2-1-0', '2-6-1', 
                    '3-3-0', '3-4-0', '3-5-0', '3-6-0', '0-4-1', '2-5-1', 
                    '3-2-0', '1-5-1', '2-0-0']
            full_seq = 'ITTITITTTV'

        seq_idx = dict(zip(seq_lst, range(len(seq_lst))))
        overfitting_scalar = [1]*len(seq_lst)
        decrease_step_cnt = dict(zip(seq_lst, [0]*len(seq_lst)))
        marked = set([])

        last_val_accs = {}
        tr_accs_100pt_cnt = dict(zip(seq_lst, [0]*len(seq_lst)))
        
        # selected_ins_ids_by_seq = {}
        # selected_ins_ids_all = {}

        # with open(os.path.join(mm_dir, 'sample_idx_by_instance_normalized_weights_ITTIV_pT.9_35k-15k-15k'), 'rb') as f:
        #     sample_idx_by_instance = pickle.load(f)
        with open(os.path.join(mm_dir, sample_weights_fn), 'r') as f:
            predefined_sample_weights = json.load(f)

        print('creating data loaders')
        dl_lst, val_dl_lst, test_dl_lst = dl.mm_batchgen(full_seq, seq_lst, 
            mm_image_dir, mm_text_dir, mm_video_dir, mm_video_process_dir,
            predefined_sample_weights, sample_idx, 
            num_workers=n_workers, batch_size=batch_size, 
            norm_weights=norm_weights, scale_weights=scale_weights,
            pick_sample_by_intsance_id=pick_sample_by_intsance_id,
            stop_overfitted_ins=stop_overfitted_ins, 
            data_seed=args.data_seed)


        print('created data loaders')
        tr_seq_sizes = [len(dataset) for dataset in dl_lst]
        print('tr_seq_sizes:', tr_seq_sizes)
        # [11368, 9275, 2542, 2893, 3693, 79, 126, 248, 382, 28, 555, 54, 8, 4, 5, 2] for batch_size 5

        val_seq_sizes = [len(dataset) for dataset in val_dl_lst]
        print('val_seq_sizes:', val_seq_sizes)
        # [4864, 3989, 1099, 1240, 1604, 32, 47, 107, 159, 10, 238, 25, 3, 3, 3, 1] for batch_size 5

        test_seq_sizes = [len(dataset) for dataset in test_dl_lst]
        print('test_seq_sizes:', test_seq_sizes)
        # [4854, 3948, 1110, 1258, 1616, 38, 56, 103, 166, 11, 240, 25, 3, 1, 2, 2] for batch_size 5

        # 0.4s/it for batch_size 8

        # randomly pick sequences
        if pick_seq == 'fn':
            with open(os.path.join(mm_dir, prob_fn), 'r') as f:
                prob_dict = json.load(f)
            p = np.array([prob_dict[s] for s in seq_lst])
        elif pick_seq == 'prob': #not random_seq:
            print('pick seq based on seq sizes')
            # pick samples from large sequence more frequently
            p = np.array(tr_seq_sizes)/sum(tr_seq_sizes)
        else:
            p = np.ones(len(seq_lst))

        # print('p:', p)
        print('prob:', dict(zip(seq_lst, p)))

        DLS = [iter(cycle(tr_dl)) for tr_dl in dl_lst]
        dl_ids = iter(cycle(range(len(dl_lst))))

        # if pick_seq == 'sequential':
        #     DLS = iter(cycle(DLS))
        #     seq_it = iter(cycle(seq_lst))

        # DLS = iter(cycle(dl_lst))
        # seq_it = iter(cycle(seq_lst))

        optimizer = ScheduledOptim(
            Adam(
                filter(lambda x: x.requires_grad, shared_model.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            512, args.n_warmup_steps,restore,max_lr=0.0001,init_lr=args.init_lr)
    elif task == 'mm_IT':
        seq_lst = ['0-1', '1-0', '1-1', '2-1', '1-2', '2-0', '2-2', '0-2']
        full_seq = 'ITIT'
        with open(os.path.join(mm_dir, 'sample_idx_ITIT_p.6.6_100000'), 'rb') as f:
            sample_idx = pickle.load(f)

        dl_lst, val_dl_lst = dl.mm_batchgen(full_seq, seq_lst, mm_image_dir, mm_text_dir, mm_video_dir, mm_video_process_dir, sample_idx, num_workers=n_workers, batch_size=batch_size)

        DLS = iter(cycle(dl_lst))
        seq_it = iter(cycle(seq_lst))

        optimizer = ScheduledOptim(
            Adam(
                filter(lambda x: x.requires_grad, shared_model.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            512, args.n_warmup_steps,restore,max_lr=0.0001,init_lr=args.init_lr)
    elif task == 'mm_vqa':
        DL0,DL1,DL2,DL3,val_dl = dl.mm_vqa_batchgen(vqa_dir, coco_images, num_workers=n_workers, batch_size=batch_size)
        DLS = iter(cycle([DL0,DL1,DL2,DL3]))
        # DL, val_dl = dl.mm_vqa_batchgen(vqa_dir, coco_images, num_workers=0, batch_size=batch_size)

        # DL,val_dl = dl.mm_batchgen(mm_dir, num_workers=0, batch_size=batch_size)
        optimizer = ScheduledOptim(
            Adam(
                filter(lambda x: x.requires_grad, shared_model.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            512, args.n_warmup_steps,restore,max_lr=0.0001,init_lr=args.init_lr)
    
    elif task == 'hmdb':
        DL,val_dl=dl.hmdb_batchgen(hmdb_data_dir,hmdb_process_dir,num_workers=n_workers,batch_size=batch_size,
                                   test_batch_size=int(batch_size/4),
                                   clip_len=16)
        optimizer = ScheduledOptim(
            Adam(
                filter(lambda x: x.requires_grad, shared_model.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            512, args.n_warmup_steps,restore,max_lr=0.0001,init_lr=args.init_lr)
    elif task == 'penn':
        DL,val_dl,test_dl=dl.penn_dataloader(penn_data_dir,batch_size=batch_size,
                                             test_batch_size=int(batch_size/2),num_workers=n_workers,vocab_file='conf/penn_vocab.json')
        optimizer = ScheduledOptim(
            Adam(
                filter(lambda x: x.requires_grad, shared_model.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            512, args.n_warmup_steps,restore,init_lr=args.init_lr)
        
    if restore != 0:
        if save_best:
            optimizer.restore(move_out_path, 'last/0')
        else:
            optimizer.restore(move_out_path, restore)

    model=model.train()

    best_val_reward = 0

    if os.path.exists(move_out_path + '/best/acc.pkl'):
        with open(move_out_path + '/best/acc.pkl', 'rb') as f:
            acc = pickle.load(f)
        best_val_acc = acc['best_val_acc']
    else:
        best_val_acc = 0
    
    if test:
        best_iteration = restore
        train_steps = start + 2
    else:
        best_iteration = save_interval
    log_str = ''

    # save_first_batch = True
    if restore != 0:
        if task == 'vqa' or task == 'vqa_struct':
            tr_dl.dataset.resume_on()
            val_dl.dataset.resume_on()
            for i in range(1, restore+2):
                if evaluating(log, eval_interval, i):
                    if i == (restore+1) and eval_first:
                        continue
                    for b in val_dl:
                        pass #print(i, 'val', b['ques'][0])
                    continue
                batch = next(DL)
                #print(i, batch['ques'][0])
                #print(i, batch['img'][0,0,5,:10], batch['img'][-1,0,5,-10:])
            #n_epochs = (restore+1)//save_interval
            #for i in range(n_epochs*(save_interval-1)):
            #    batch = next(DL)
            #    print(batch['ques'][0])

            # with open(model_save_path+'_%s_last_batch_questions_restore.pkl'%restore, 'wb') as f:
            #     pickle.dump(batch['ques'], f)
            val_dl.dataset.resume_off()
            tr_dl.dataset.resume_off()

    for i in range(start+1, train_steps):
        model.zero_grad()
        if barrier is not None:
            barrier.wait()
        if gpu_id > 0:
            with torch.cuda.device(gpu_id):
                model.load_state_dict(shared_model.state_dict())
                       
        # Calculate loss
        step = counter.increment()

        if task == 'mm_ITV' or task == 'mm_seq10':
            if not notest and i + 1 >= train_steps:
                # pretrained_dict=torch.load(os.path.join(model_save_path, 'best_val_model.pth'))
                # model_dict=shared_model.state_dict()
                # pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                #                (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
                # model.load_state_dict(pretrained_dict,strict=False)
                # print(best_iteration)
                if save_best:
                    shared_model.restore(move_out_path, 'best/0')
                    optimizer.restore(move_out_path, 'best/0')
                else:
                    shared_model.restore(move_out_path, best_iteration)
                    optimizer.restore(move_out_path, best_iteration)

                log_str += 'Restored existing model with iterations: %d\n' % (best_iteration)
                model = shared_model
                model = model.eval()
                print('-' * 100)
                print('Test model')
                log_str += '-' * 100 + '\nTest model\n'

                total = 0
                total_correct = 0
                total_reward = 0
                for seq, val_dl in zip(seq_lst, test_dl_lst):
                    print(seq)
                    log_str += '{}\n'.format(seq)
                    val_loss = 0
                    loss_total = 0
                    val_correct = 0
                    val_total = 0
                    val_reward = 0
                    # j=0
                    for b in val_dl: #tqdm(val_dl):
                        # j += 1
                        # if j > 5:
                        #     break
                        imgs = b['imgs']
                        text = b['text']
                        videos = b['videos']
                        sample_weights = b['sample_weights']
                        labels = b['labels']
                        structured = None
                        # structured = b['struct']
                        if gpu_id >= 0:
                            imgs = send_to_device(imgs, gpu_id) #imgs.cuda(device=gpu_id)
                            # text = send_to_device(text, gpu_id)
                            videos = send_to_device(videos, gpu_id)
                            # structured = send_to_device(structured)
                            labels = send_to_device(labels, gpu_id) #labels.cuda(device=gpu_id)
                            sample_weights = send_to_device(sample_weights, gpu_id)
                            # answers=answers.cuda(device=gpu_id)
                        pred, loss, acc, n_correct, n_total = r.mm(model, imgs, text, videos, structured, targets=labels, mode='val',return_str_preds=True)
                        val_loss += float(loss.detach().cpu().numpy())
                        loss_total += 1
                        # val_acc+=acc
                        val_correct += n_correct
                        val_total += n_total

                    val_loss/=loss_total #len(val_dl)
                    # val_acc=(val_acc/len(val_dl))

                    total_correct += val_correct
                    total += val_total

                    val_acc = (val_correct/val_total)*100
                    if rewarding:
                        val_reward = reward_ITTIV(seq, val_acc/100)
                        total_reward += val_reward

                    summary_writer.add_scalar('Test_loss_%s'%seq, val_loss, step)
                    print('Step %d, %s, mm test loss: %f, Accuracy %f %%, reward: %f' % (step, seq, val_loss, val_acc, val_reward))
                    log_str += 'Step %d, %s, mm test loss: %f, Accuracy %f %%, reward: %f\n' % (step, seq, val_loss, val_acc, val_reward)

                print('-' * 100)
                log_str += '-' * 100 + '\n'
                total_val_acc = (total_correct/total)*100
                print('Step %d, mm total test Accuracy %f %%, reward: %f'%(step, total_val_acc, total_reward))
                log_str += 'Step %d, mm total test Accuracy %f %%, reward: %f\n'%(step, total_val_acc, total_reward)

            if not test and evaluating(log, eval_interval, i):
                if i == (start+1) and not eval_first:
                    continue
            # if (log and eval_interval is not None and i % eval_interval == 0 and ((i == start and eval_first) or i > start))):
                model = model.eval()
                print('-' * 100)
                print('Evaluation step')
                log_str += '-' * 100 + '\nEvaluation step\n'

                total = 0
                total_correct = 0
                total_reward = 0
                for seq, val_dl in zip(seq_lst, val_dl_lst):
                    if skip_seqs is not None and seq in skip_seqs:
                        continue

                    val_loss = 0
                    loss_total = 0
                    val_correct = 0
                    val_total = 0
                    val_reward = 0
                    
                    val_dl_size = len(val_dl)
                    # # j=0

                    # n_selected = val_dl_size
                    # if n_selected >= 50:
                    #     n_selected = val_dl_size//2

                    # selected_val_mini_batch_ids = np.random.choice(range(val_dl_size), n_selected, replace=False)

                    # print(seq, 'val size:', n_selected)
                    # log_str += '{} val size: {}\n'.format(seq, n_selected)

                    print(seq, 'val size:', val_dl_size)
                    log_str += '{} val size: {}\n'.format(seq, val_dl_size)
                    start_time = time.time()
                    
                    for b in val_dl: #for bi,b in enumerate(val_dl):
                        # j+=1
                        # if j > 5:
                        #     break
                        # if bi not in selected_val_mini_batch_ids:
                        #     continue

                        imgs = b['imgs']
                        text = b['text']
                        videos = b['videos']
                        sample_weights = b['sample_weights']
                        labels = b['labels']
                        structured = None
                        # structured = b['struct']
                        if gpu_id >= 0:
                            imgs = send_to_device(imgs, gpu_id) #imgs.cuda(device=gpu_id)
                            # text = send_to_device(text, gpu_id)
                            videos = send_to_device(videos, gpu_id)
                            # structured = send_to_device(structured)
                            labels = send_to_device(labels, gpu_id) #labels.cuda(device=gpu_id)
                            sample_weights = send_to_device(sample_weights, gpu_id)
                            # answers=answers.cuda(device=gpu_id)
                        pred, loss, acc, n_correct, n_total = r.mm(model, imgs, text, videos, structured, targets=labels, mode='val',return_str_preds=True)
                        val_loss += float(loss.detach().cpu().numpy())
                        loss_total += 1
                        # val_acc+=acc
                        val_correct += n_correct
                        val_total += n_total

                    val_loss/=loss_total #len(val_dl)
                    # val_acc=(val_acc/len(val_dl))
                    total_correct += val_correct
                    total += val_total

                    val_acc = (val_correct/val_total)

                    if rewarding:
                        val_reward = reward_ITTIV(seq, val_acc)
                        total_reward += val_reward

                    if step > overfitting_start and overfitting(seq, last_val_accs, val_acc, decrease_step_cnt, overfitting_threshold):
                        print('model is overfitted on %s'%seq)
                        log_str += 'model is overfitted on %s: validation acc drops in 2 consecutive val steps\n'%seq
                        overfitting_scalar[seq_idx[seq]] = 0

                        if stop_overfitted_ins:
                            ins_ids = dl_lst[seq_idx[seq]].dataset.get_ins_ids()
                            for ins_id in ins_ids:
                                if ins_id not in marked:
                                    marked.add(ins_id)

                            for dl_id in range(len(dl_lst)):
                                dl_lst[dl_id].dataset.set_marked(marked)
                                dl_lst[dl_id].dataset.remove_marked_ins()


                    last_val_accs[seq] = val_acc

                    summary_writer.add_scalar('Val_loss_%s'%seq, val_loss, step)
                    print('Step %d, %s, mm validation loss: %f, Accuracy %f %%, reward: %f' % (step, seq, val_loss,val_acc*100,val_reward))
                    log_str += 'Step %d, %s, mm validation loss: %f, Accuracy %f %%, reward: %f\n' % (step, seq, val_loss,val_acc*100,val_reward)
                    end_time = time.time()
                    print('Step {}, {}, validation takes {:.2f}s\n'.format(step, seq, end_time - start_time))
                    log_str += 'Step {}, {}, validation takes {:.2f}s\n'.format(step, seq, end_time - start_time)
                    

                print('-' * 100)
                log_str += '-' * 100 + '\n'

                total_val_acc = (total_correct/total)*100

                if rewarding and total_reward > best_val_reward:
                    best_val_reward = total_reward
                    best_iteration = step-1
                    print(best_iteration)
                    log_str += 'best_iteration:{}\n'.format(best_iteration)

                    if save_best:
                        shared_model.save(move_out_path, 'best/0')
                        optimizer.save(move_out_path, 'best/0')

                if not rewarding and total_val_acc > best_val_acc:
                    best_val_acc = total_val_acc
                    best_iteration = step-1
                    print(best_iteration)
                    log_str += 'best_iteration:{}\n'.format(best_iteration)

                    if save_best:
                        shared_model.save(move_out_path, 'best/0')
                        optimizer.save(move_out_path, 'best/0')
                    # best_model = shared_model.state_dict()
                    # shared_model.save(model_save_path, step, best_model=True)
                print('Step %d, mm total validation Accuracy %f %%, reward: %f'%(step, total_val_acc, total_reward))
                log_str += 'Step %d, mm total validation Accuracy %f %%, reward: %f\n'%(step, total_val_acc, total_reward)

                with open(move_out_path + '.log', 'a') as f: #open(os.path.join(model_save_path, 'log.txt'), 'a') as f:
                    print(log_str, file=f)
                    log_str = ''

                # skip training and validation when all instances 
                # are removed due to overfitting
                if stop_overfitted_ins:
                    empty_seq_cnt = 0
                    for dl_id in range(len(dl_lst)):
                        if dl_lst[dl_id].dataset.empty():
                            empty_seq_cnt += 1

                    if empty_seq_cnt == len(dl_lst):
                        test = True

                model = model.train()
                continue

            if not test:
                prob = p*overfitting_scalar
                prob = prob/sum(prob)

                if pick_seq == 'sequential':
                    next_dl_id = next(dl_ids)
                    # DL = next(DLS)
                    # seq = next(seq_it)
                else:
                    next_dl_id = np.random.choice(range(len(p)),p=prob)
                
                DL = DLS[next_dl_id]
                seq = seq_lst[next_dl_id]

                if skip_seqs is not None and seq in skip_seqs:
                    continue

                # Stop training on instances, which contains samples 
                # that belong to overfitted sequence type
                if stop_overfitted_ins and dl_lst[next_dl_id].dataset.empty():
                    continue
                
                batch = next(DL)
                

                # selected_ins_ids = dl_lst[next_dl_id].dataset.get_selected_ins_ids()
                # if seq not in selected_ins_ids_by_seq:
                #     selected_ins_ids_by_seq[seq] = {}

                # for ins_id in selected_ins_ids:
                #     if ins_id in selected_ins_ids_by_seq[seq]:
                #         selected_ins_ids_by_seq[seq][ins_id] += 1
                #     else:
                #         selected_ins_ids_by_seq[seq][ins_id] = 1

                #     if ins_id in selected_ins_ids_all:
                #         selected_ins_ids_all[ins_id] += 1
                #     else:
                #         selected_ins_ids_all[ins_id] = 1


                # print('seq %s: '%seq, selected_ins_ids[:5])

                if pick_sample_by_intsance_id:
                    marked = dl_lst[next_dl_id].dataset.get_marked()
                    for dl_id in range(len(dl_lst)):
                        dl_lst[dl_id].dataset.set_marked(marked)

                # DL = DLS[np.random.randint(4)]
                # DL = next(DLS)
                # batch = next(DL)
                if gpu_id >= 0:
                    imgs = send_to_device(batch['imgs'], gpu_id)
                    videos = send_to_device(batch['videos'], gpu_id)
                    labels = send_to_device(batch['labels'], gpu_id)
                    sample_weights = send_to_device(batch['sample_weights'], gpu_id)
                else:
                    imgs = batch['imgs']
                    videos = batch['videos']
                    labels = batch['labels']
                    sample_weights = batch['sample_weights']
                
                text = batch['text']
                    # videos = batch['videos']
                    # structured = batch['struct']
                
                # videos=None
                structured=None
                    
                _, loss, acc, _, _ = r.mm(model, imgs, text, videos, structured, targets=labels, sample_weights=sample_weights)
                loss.backward()
                loss=loss.detach()

                if detect_overfitting_with_tr_acc and acc == 100:
                    tr_accs_100pt_cnt[seq] += 1

                    if tr_accs_100pt_cnt[seq] >= 5:
                        print('model is overfitted on %s'%seq)
                        log_str += 'model is overfitted on %s: 100 tr acc for 5 iters\n'%seq
                        overfitting_scalar[seq_idx[seq]] = 0

                        if stop_overfitted_ins:
                            ins_ids = dl_lst[seq_idx[seq]].dataset.get_ins_ids()
                            for ins_id in ins_ids:
                                if ins_id not in marked:
                                    marked.add(ins_id)

                            for dl_id in range(len(dl_lst)):
                                dl_lst[dl_id].dataset.set_marked(marked)
                                dl_lst[dl_id].dataset.remove_marked_ins()
                else:
                    tr_accs_100pt_cnt[seq] = 0

                if log:
                    summary_writer.add_scalar('Loss_%s'%seq, loss, step)
                print('Step %d, %s, mm Loss: %f, Accuracy:  %f %%' % (step, seq, loss,acc))
                log_str += 'Step %d, %s, mm Loss: %f, Accuracy:  %f %%\n' % (step, seq, loss,acc)

        elif task == 'mm_IT':
            if evaluating(log, eval_interval, i):
                if i == (start+1) and not eval_first:
                    continue
            # if (log and eval_interval is not None and i % eval_interval == 0 and i > 0):
                model = model.eval()
                print('-' * 100)
                print('Evaluation step')
                for seq, val_dl in zip(seq_lst, val_dl_lst):
                    val_loss = 0
                    val_acc = 0
                    for b in tqdm(val_dl):
                        imgs = b['imgs']
                        labels = b['labels']
                        text = b['text']
                        sample_weights = b['sample_weights']
                        videos = None
                        structured = None
                        # videos = b['videos']
                        # structured = b['struct']
                        if gpu_id >= 0:
                            imgs = send_to_device(imgs, gpu_id) #imgs.cuda(device=gpu_id)
                            # text = send_to_device(text, gpu_id)
                            # videos = send_to_device(videos)
                            # structured = send_to_device(structured)
                            labels = send_to_device(labels, gpu_id) #labels.cuda(device=gpu_id)
                            sample_weights = send_to_device(sample_weights, gpu_id)
                            # answers=answers.cuda(device=gpu_id)
                        pred, loss,acc = r.mm(model, imgs, text, videos, structured, targets=labels, mode='val',return_str_preds=True)
                        val_loss += float(loss.detach().cpu().numpy())
                        val_acc+=acc
                    val_loss/=len(val_dl)
                    val_acc=(val_acc/len(val_dl))
                    summary_writer.add_scalar('Val_loss_%s'%seq, val_loss, step)
                    print('Step %d, %s, mm validation loss: %f, Accuracy %f %%' % (step, seq, val_loss,val_acc))
                print('-' * 100)
                model = model.train()
                continue
            # DL = DLS[np.random.randint(4)]
            DL = next(DLS)
            batch = next(DL)
            if gpu_id >= 0:
                imgs = send_to_device(batch['imgs'], gpu_id)
                labels = send_to_device(batch['labels'], gpu_id)
                sample_weights = send_to_device(batch['sample_weights'], gpu_id)
            else:
                imgs = batch['imgs']
                labels = batch['labels']
                sample_weights = batch['sample_weights']
            
            text = batch['text']
                # videos = batch['videos']
                # structured = batch['struct']
            
            videos=None
            structured=None
                
            _, loss,acc = r.mm(model, imgs, text, videos, structured, targets=labels, sample_weights=sample_weights)
            loss.backward()
            loss=loss.detach()

            seq = next(seq_it)
            if log:
                summary_writer.add_scalar('Loss_%s'%seq, loss, step)
            print('Step %d, %s, mm Loss: %f, Accuracy:  %f %%' % (step, seq, loss,acc))

        elif task == 'mm_vqa':
            if evaluating(log, eval_interval, i):
                if i == (start+1) and not eval_first:
                    continue
            # if (log and eval_interval is not None and i % eval_interval == 0 and i > 0):
                model = model.eval()
                val_loss = 0
                val_acc=0
                print('-' * 100)
                print('Evaluation step')
                for b in tqdm(val_dl):
                    imgs = b['img']
                    labels = b['ans']
                    text = b['ques']
                    sample_weights = b['sample_weights']
                    videos = None
                    structured = None
                    # videos = b['videos']
                    # structured = b['struct']
                    if gpu_id >= 0:
                        imgs = send_to_device(imgs, gpu_id) #imgs.cuda(device=gpu_id)
                        # text = send_to_device(text, gpu_id)
                        # videos = send_to_device(videos)
                        # structured = send_to_device(structured)
                        labels = send_to_device(labels, gpu_id) #labels.cuda(device=gpu_id)
                        sample_weights = send_to_device(sample_weights, gpu_id)
                        # answers=answers.cuda(device=gpu_id)
                    pred, loss,acc = r.mm(model, imgs, text, videos, structured, targets=labels, mode='val',return_str_preds=True, sample_weights=sample_weights)
                    val_loss += float(loss.detach().cpu().numpy())
                    val_acc+=acc
                val_loss/=len(val_dl)
                val_acc=(val_acc/len(val_dl))
                summary_writer.add_scalar('Val_loss', val_loss, step)
                print('Step %d, mm validation loss: %f, Accuracy %f %%' % (step, val_loss,val_acc))
                print('-' * 100)
                model = model.train()
                continue
            # DL = DLS[np.random.randint(4)]
            DL = next(DLS)
            batch = next(DL)
            if gpu_id >= 0:
                imgs = send_to_device(batch['img'], gpu_id) #batch['img'].cuda(device=gpu_id)
                # text = send_to_device(batch['ques'], gpu_id) # batch['text'].cuda(device=gpu_id)
                # videos = send_to_device(batch['videos']) #batch['videos'].cuda(device=gpu_id)
                # structured = send_to_device(batch['struct']) #batch['struct'].cuda(device=gpu_id)
                labels = send_to_device(batch['ans'], gpu_id) #batch['label'].cuda(device=gpu_id)
                sample_weights = send_to_device(batch['sample_weights'], gpu_id)
            else:
                imgs = batch['img']
                labels = batch['ans']
                sample_weights = batch['sample_weights']
            
            text = batch['ques']
                # videos = batch['videos']
                # structured = batch['struct']
            
            videos=None
            structured=None
            
            
                
            _, loss,acc = r.mm_vqa(model, imgs, text, videos, structured, targets=labels, sample_weights=sample_weights)
            loss.backward()
            loss=loss.detach()
            if log:
                summary_writer.add_scalar('Loss', loss, step)
            print('Step %d, mm Loss: %f, Accuracy:  %f %%' % (step, loss,acc))

        elif task == 'caption':
            if evaluating(log, eval_interval, i):
                if i == (start+1) and not eval_first:
                    continue
            # if (log and eval_interval is not None and i % eval_interval == 0 and i > 0):
                model = model.eval()
                val_loss=0
                val_acc=0
                print('-' * 100)
                print('Evaluation step')
                for b in tqdm(val_dl):
                    imgs = b['img']
                    if gpu_id>=0:
                        imgs=imgs.cuda(device=gpu_id)
                    captions = b['cap']
                    # In val mode we do not pass the targets for prediction. We use it only for loss calculation
                    _,loss,acc = r.image_caption(model, imgs, targets=captions, mode='val',return_str_preds=True)
                    val_loss += float(loss.detach().cpu().numpy())
                    val_acc+=acc
                val_loss/=len(val_dl)
                val_acc=(val_acc/len(val_dl))
                summary_writer.add_scalar('Val_loss', val_loss, step)
                print('Step %d, COCO validation loss: %f, Accuracy %f %%' % (step, val_loss,val_acc))
                print('-' * 100)
                model = model.train()
            batch = next(DL)
            if gpu_id >= 0:
                imgs = batch['img'].cuda(device=gpu_id)
            else:
                imgs = batch['img']
            captions = batch['cap']
            _, loss,acc = r.image_caption(model, imgs, targets=captions)
            loss.backward()
            loss=loss.detach()
            if log:
                summary_writer.add_scalar('Loss', loss, step)
            print('Step %d, Caption Loss: %f, Accuracy:  %f %%' % (step, loss,acc))
            
        elif task == 'vqa' or task == 'vqa_struct':
            if with_val and i + 1 >= train_steps:
                if save_best:
                    shared_model.restore(move_out_path, 'best/0')
                    optimizer.restore(move_out_path, 'best/0')
                else:
                    shared_model.restore(move_out_path, best_iteration)
                    optimizer.restore(move_out_path, best_iteration)

                log_str += 'Restored existing model with iterations: %d\n' % (best_iteration)
                model = shared_model
                model = model.eval()
                print('-' * 100)
                print('Test model')
                log_str += '-' * 100 + '\nTest model\n'

                val_loss = 0
                val_acc=0
                val_emb_spat_loss = 0
                val_emb_temp_loss = 0
                print('-' * 100)
                print('Evaluation step')
                start_time = time.time()
                log_str += '-' * 100 + '\nEvaluation step\n'

                bi = 0
                for b in tqdm(test_dl):
                    imgs = b['img']
                    answers=b['ans']
                    if gpu_id >= 0:
                        imgs = imgs.cuda(device=gpu_id)
                        answers=answers.cuda(device=gpu_id)
                    questions= b['ques']
                    
                    if task == 'vqa_struct':
                        structured = b['struct'].cuda(device=gpu_id)
                        bbox_mask = b['bbox_mask']
                        if bbox_mask is not None:
                            bbox_mask = bbox_mask.cuda(device=gpu_id)
                        # structured dim:
                        if args.fusion:
                            pred, loss, acc, emb_loss_spat, emb_loss_temp, spat_struct_enc_gate, temp_struct_enc_gate = r.vqa_fusion(model, imgs, questions, structured=structured, targets=answers, mode='val',return_str_preds=True, bbox_mask=bbox_mask)

                            if temp_struct_enc_gate is not None:
                                temp_struct_enc_gate_test[bi] = temp_struct_enc_gate.squeeze().detach().cpu().numpy()
                            if spat_struct_enc_gate is not None:
                                spat_struct_enc_gate_test[bi] = spat_struct_enc_gate.squeeze().detach().cpu().numpy()
                        else:
                            pred, loss, acc = r.vqa(model, imgs, questions, structured=structured, targets=answers, mode='val',return_str_preds=True, bbox_mask=bbox_mask)

                        bi += 1
                    else:
                        # In val mode we do not pass the targets for prediction. We use it only for loss calculation
                        pred, loss, acc = r.vqa(model, imgs, questions,targets=answers, mode='val',return_str_preds=True)
                    
                    val_loss += float(loss.detach().cpu().numpy())

                    if emb_loss_spat is not None and emb_loss_temp is not None:
                        val_emb_spat_loss += float(emb_loss_spat.detach().cpu().numpy())
                        val_emb_temp_loss += float(emb_loss_temp.detach().cpu().numpy())
                    val_acc+=acc
                val_loss/=len(val_dl)
                val_emb_spat_loss/=len(val_dl)
                val_emb_temp_loss/=len(val_dl)
                val_acc=(val_acc/len(val_dl))
                summary_writer.add_scalar('Test_loss', val_loss, step)

                if task == 'vqa_struct':
                    print('Step %d, VQA test Loss: %f, Spat Emb Loss: %f, Temp Emb Loss: %f, Accuracy %f %%' % (step, val_loss, val_emb_spat_loss, val_emb_temp_loss, val_acc))
                    log_str += 'Step %d, VQA test Loss: %f, Spat Emb Loss: %f, Temp Emb Loss: %f, Accuracy %f %%\n' % (step, val_loss, val_emb_spat_loss, val_emb_temp_loss, val_acc)

                    if (args.inject_at_encoder or args.inject_after_encoder) and args.spat_fusion_attn_type != 'none':
                        with open(model_save_path+'_spat_struct_enc_gate_test.dict', 'wb') as f:
                            pickle.dump(spat_struct_enc_gate_test, f)
                    if (args.inject_at_encoder or args.inject_after_encoder) and args.temp_fusion_attn_type != 'none':
                        with open(model_save_path+'_temp_struct_enc_gate_test.dict', 'wb') as f:
                            pickle.dump(temp_struct_enc_gate_test, f)
                else:
                    print('Step %d, VQA test loss: %f, Accuracy %f %%' % (step, val_loss, val_acc))
                    log_str += 'Step %d, VQA test loss: %f, Accuracy %f %%\n' % (step, val_loss, val_acc)
                
                try:
                    with open(move_out_path + '.log', 'a') as f: #open(os.path.join(model_save_path, 'log.txt'), 'a') as f:
                        print(log_str, file=f)
                        log_str = ''
                except:
                    pass

            if evaluating(log, eval_interval, i):
                if i == (start+1) and not eval_first:
                    continue
            # if (log and eval_interval is not None and i % eval_interval == 0 and i > start):
                model = model.eval()
                val_loss = 0
                val_acc=0
                val_emb_spat_loss = 0
                val_emb_temp_loss = 0
                print('-' * 100)
                print('Evaluation step')
                start_time = time.time()
                log_str += '-' * 100 + '\nEvaluation step\n'

                bi = 0
                for b in tqdm(val_dl):
                    imgs = b['img']
                    answers=b['ans']
                    if gpu_id >= 0:
                        imgs = imgs.cuda(device=gpu_id)
                        answers=answers.cuda(device=gpu_id)
                    questions= b['ques']
                    # print(questions[0])
                    
                    if task == 'vqa_struct':
                        structured = b['struct'].cuda(device=gpu_id)
                        bbox_mask = b['bbox_mask']
                        if bbox_mask is not None:
                            bbox_mask = bbox_mask.cuda(device=gpu_id)

                        # structured dim:
                        if args.fusion:
                            pred, loss, acc, emb_loss_spat, emb_loss_temp, spat_struct_enc_gate, temp_struct_enc_gate = r.vqa_fusion(model, imgs, questions, structured=structured, targets=answers, mode='val',return_str_preds=True,bbox_mask=bbox_mask)

                            if temp_struct_enc_gate is not None:
                                temp_struct_enc_gate_val[bi] = temp_struct_enc_gate.squeeze().detach().cpu().numpy()
                            if spat_struct_enc_gate is not None:
                                spat_struct_enc_gate_val[bi] = spat_struct_enc_gate.squeeze().detach().cpu().numpy()
                        else:
                            pred, loss, acc = r.vqa(model, imgs, questions, structured=structured, targets=answers, mode='val',return_str_preds=True,bbox_mask=bbox_mask)

                        bi += 1
                    else:
                        # In val mode we do not pass the targets for prediction. We use it only for loss calculation
                        pred, loss, acc = r.vqa(model, imgs, questions,targets=answers, mode='val',return_str_preds=True)
                    
                    val_loss += float(loss.detach().cpu().numpy())

                    if emb_loss_spat is not None and emb_loss_temp is not None:
                        val_emb_spat_loss += float(emb_loss_spat.detach().cpu().numpy())
                        val_emb_temp_loss += float(emb_loss_temp.detach().cpu().numpy())
                    val_acc+=acc
                val_loss/=len(val_dl)
                val_emb_spat_loss/=len(val_dl)
                val_emb_temp_loss/=len(val_dl)
                val_acc=(val_acc/len(val_dl))
                summary_writer.add_scalar('Val_loss', val_loss, step)

                if task == 'vqa_struct':
                    print('Step %d, VQA validation Loss: %f, Spat Emb Loss: %f, Temp Emb Loss: %f, Accuracy %f %%' % (step, val_loss, val_emb_spat_loss, val_emb_temp_loss, val_acc))
                    log_str += 'Step %d, VQA validation Loss: %f, Spat Emb Loss: %f, Temp Emb Loss: %f, Accuracy %f %%\n' % (step, val_loss, val_emb_spat_loss, val_emb_temp_loss, val_acc)

                    if (args.inject_at_encoder or args.inject_after_encoder) and args.spat_fusion_attn_type != 'none':
                        with open(model_save_path+'_spat_struct_enc_gate_val_%s.dict'%i, 'wb') as f:
                            pickle.dump(spat_struct_enc_gate_val, f)
                    if (args.inject_at_encoder or args.inject_after_encoder) and args.temp_fusion_attn_type != 'none':
                        with open(model_save_path+'_temp_struct_enc_gate_val_%s.dict'%i, 'wb') as f:
                            pickle.dump(temp_struct_enc_gate_val, f)
                else:
                    print('Step %d, VQA validation loss: %f, Accuracy %f %%' % (step, val_loss, val_acc))
                    log_str += 'Step %d, VQA validation loss: %f, Accuracy %f %%\n' % (step, val_loss, val_acc)
                
                print('-' * 100)
                log_str += '-' * 100 + '\n'
                end_time = time.time()
                print('Validation takes {:.2f}s\n'.format(end_time - start_time))
                log_str += 'Validation takes {:.2f}s\n'.format(end_time - start_time)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_iteration = step-1
                    print(best_iteration)
                    log_str += 'best_iteration:{}\n'.format(best_iteration)

                    shared_model.save(move_out_path, 'best/0')
                    optimizer.save(move_out_path, 'best/0')

                    with open(move_out_path + '/best/acc.pkl', 'wb') as f:
                        pickle.dump({'best_val_acc': best_val_acc, 'best_iteration': best_iteration}, f)

                model = model.train()

                # save_first_batch = True

                try:
                    with open(move_out_path + '.log', 'a') as f: #open(os.path.join(model_save_path, 'log.txt'), 'a') as f:
                        print(log_str, file=f)
                        log_str = ''
                except:
                    pass

                if evaluate:
                    return

                continue

            # start_time = time.time()
            batch = next(DL)
            # print(batch['ques'][:5])
            #print(batch['img'][0,0,5,:10],batch['img'][-1,0,5,-10:])

            # if save_first_batch:
            #     if restore != 0:
            #         with open(model_save_path+'_%s_first_batch_questions_restore.pkl'%i, 'wb') as f:
            #             pickle.dump(batch['ques'], f)
            #     else:
            #         with open(model_save_path+'_%s_first_batch_questions.pkl'%i, 'wb') as f:
            #             pickle.dump(batch['ques'], f)

            #     save_first_batch = False

            # end_time = time.time()
            # log_str += 'fetching next batch takes {:.2f}s\n'.format(end_time - start_time)
            # print('fetching next batch takes {:.2f}s'.format(end_time - start_time))
            
            # start_time = time.time()
            if gpu_id >= 0:
                imgs = batch['img'].cuda(device=gpu_id)
                answers = batch['ans'].cuda(device=gpu_id)
            else:
                imgs = batch['img']
                answers = batch['ans']
            questions = batch['ques']
            # end_time = time.time()
            # log_str += 'send next batch to gpu takes {:.2f}s\n'.format(end_time - start_time)
            # print('send next batch gpu takes {:.2f}s'.format(end_time - start_time))
            # print(questions[0], questions[-1])
            # print(imgs[0,0,5,:10], imgs[-1,0,5,-10:])
            # print(answers[0], answers[-1])

            # start_time = time.time()
            if task == 'vqa_struct':
                structured = batch['struct'].cuda(device=gpu_id)
                bbox_mask = batch['bbox_mask']
                if bbox_mask is not None:
                    bbox_mask = bbox_mask.cuda(device=gpu_id)

                if args.fusion:
                    _, loss, acc, emb_loss_spat, emb_loss_temp, spat_struct_enc_gate, temp_struct_enc_gate = r.vqa_fusion(model, imgs, questions, structured=structured, targets=answers, bbox_mask=bbox_mask)
                
                    # print(temp_struct_enc_gate.shape)
                    # print('temp_struct_enc_gate:', temp_struct_enc_gate.squeeze().detach().cpu().numpy())
                    # print(spat_struct_enc_gate.shape)
                    # print('spat_struct_enc_gate:', spat_struct_enc_gate.squeeze().detach().cpu().numpy())

                    if temp_struct_enc_gate is not None:
                        temp_struct_enc_gate_tr[(i-1)%eval_interval] = temp_struct_enc_gate.squeeze().detach().cpu().numpy()
                    if spat_struct_enc_gate is not None:
                        spat_struct_enc_gate_tr[(i-1)%eval_interval] = spat_struct_enc_gate.squeeze().detach().cpu().numpy()

                    if emb_loss_spat is not None and emb_loss_temp is not None:
                        loss = loss + emb_loss_spat + emb_loss_temp
                    # emb_loss = r.vqa_struct_emb(model, imgs, questions, structured)
                    # total_loss = loss + emb_loss
                else:
                    _, loss, acc = r.vqa(model, imgs, questions, structured=structured, targets=answers, bbox_mask=bbox_mask)

            else:
                _, loss, acc = r.vqa(model, imgs, questions, targets=answers)
            # end_time = time.time()
            # log_str += 'forward pass takes {:.2f}s\n'.format(end_time - start_time)
            # print('forward pass takes {:.2f}s'.format(end_time - start_time))

            # start_time = time.time()
            # emb_loss.backward(retain_graph=True)
            # end_time = time.time()
            # log_str += 'emb loss backward pass takes {:.2f}s\n'.format(end_time - start_time)
            # print('emb loss backward pass takes {:.2f}s'.format(end_time - start_time))

            # print(loss.detach())

            # start_time = time.time()
            loss.backward()
            # end_time = time.time()
            # log_str += 'backward pass takes {:.2f}s\n'.format(end_time - start_time)
            # print('backward pass takes {:.2f}s'.format(end_time - start_time))

            loss=loss.detach()
            # emb_loss=emb_loss.detach()
            if log:
                summary_writer.add_scalar('Loss', loss, step)

            # if task == 'vqa_struct':
            if emb_loss_spat is not None and emb_loss_temp is not None:
                print('Step %d, VQA Loss: %f, Spat Emb Loss: %f, Temp Emb Loss: %f, Accuracy:  %f %%' % (step, loss, emb_loss_spat.detach(), emb_loss_temp.detach(), acc))
                log_str += 'Step %d, VQA Loss: %f, Spat Emb Loss: %f, Temp Emb Loss: %f, Accuracy:  %f %%\n' % (step, loss, emb_loss_spat.detach(), emb_loss_temp.detach(), acc)
            else:
                print('Step %d, VQA Loss: %f, Accuracy:  %f %%' % (step, loss, acc))
                log_str += 'Step %d, VQA Loss: %f, Accuracy:  %f %%\n' % (step, loss, acc)

            
        elif task == 'birds' or task == 'birds_struct':
            if not test and evaluating(log, eval_interval, i):
                if i == (start+1) and not eval_first:
                    continue
            # if (log and eval_interval is not None and i % eval_interval == 0 and i > start):
                model = model.eval()
                val_loss = 0
                val_acc=0
                print('-' * 100)
                print('Evaluation step')
                log_str += '-' * 100 + '\nEvaluation step\n'
                for b in tqdm(val_dl):
                    imgs = b['img']
                    labels = b['label']
                    if gpu_id >= 0:
                        imgs = imgs.cuda(device=gpu_id)
                        labels = labels.cuda(device=gpu_id)
                        # answers=answers.cuda(device=gpu_id)
                    if task == 'birds_struct':
                        structured = b['struct']
                        structured = structured.cuda(device=gpu_id)
                        # In val mode we do not pass the targets for prediction. We use it only for loss calculation
                        pred, loss,acc = r.birds(model, imgs, structured=structured, targets=labels, mode='val',return_str_preds=True)
                    else:
                        pred, loss,acc = r.birds(model, imgs, targets=labels, mode='val',return_str_preds=True)
                    val_loss += float(loss.detach().cpu().numpy())
                    val_acc+=acc
                val_loss/=len(val_dl)
                val_acc=(val_acc/len(val_dl))
                summary_writer.add_scalar('Val_loss', val_loss, step)
                print('Step %d, birds validation loss: %f, Accuracy %f %%' % (step, val_loss,val_acc))
                print('-' * 100)

                log_str += 'Step %d, birds validation loss: %f, Accuracy %f %%\n' % (step, val_loss,val_acc)
                log_str += '-' * 100 + '\n'

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_iteration = step-1
                    print(best_iteration)
                    log_str += 'best_iteration:{}\n'.format(best_iteration)

                    if save_best:
                        shared_model.save(move_out_path, 'best/0')
                        optimizer.save(move_out_path, 'best/0')  

                model = model.train()
                continue

            if not test:
                batch = next(DL)
                if gpu_id >= 0:
                    imgs = batch['img'].cuda(device=gpu_id)
                    labels = batch['label'].cuda(device=gpu_id)
                    # answers = batch['ans'].cuda(device=gpu_id)
                else:
                    imgs = batch['img']
                    labels = batch['label']
                    # answers = batch['ans']
                # labels = batch['label']
                # print(labels.shape)
                # print(imgs.shape)
                if task == 'birds_struct':
                    structured = batch['struct']
                    structured = structured.cuda(device=gpu_id)  
                    # structured = torch.empty(batch_size, 512, 512).uniform_(0,1).cuda(device=gpu_id)                  
                    _, loss,acc = r.birds(model, imgs, structured=structured, targets=labels)
                else:
                    _, loss,acc = r.birds(model, imgs, targets=labels)
                loss.backward()
                loss=loss.detach()
                if log:
                    summary_writer.add_scalar('Loss', loss, step)
                print('Step %d, birds Loss: %f, Accuracy:  %f %%' % (step, loss,acc))
                log_str += 'Step %d, birds Loss: %f, Accuracy:  %f %%\n' % (step, loss,acc)

            if not notest and i + 1 >= train_steps:
                # pretrained_dict=torch.load(os.path.join(model_save_path, 'best_val_model.pth'))
                # model_dict=shared_model.state_dict()
                # pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                #                (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
                # model.load_state_dict(pretrained_dict,strict=False)
                # print(best_iteration)
                if not testing:
                    shared_model.restore(move_out_path, 'best')
                    optimizer.restore(move_out_path, 'best')
                    log_str += 'Restored existing model with iterations: %d\n' % (best_iteration)
                    model = shared_model
                model = model.eval()
                print('-' * 100)
                print('Test model')
                log_str += '-' * 100 + '\nTest model\n'

                val_loss = 0
                val_acc=0
                for b in tqdm(test_dl):
                    imgs = b['img']
                    labels = b['label']
                    if gpu_id >= 0:
                        imgs = imgs.cuda(device=gpu_id)
                        labels = labels.cuda(device=gpu_id)
                        # answers=answers.cuda(device=gpu_id)
                    if task == 'birds_struct':
                        structured = b['struct']
                        structured = structured.cuda(device=gpu_id)
                        # In val mode we do not pass the targets for prediction. We use it only for loss calculation
                        pred, loss,acc = r.birds(model, imgs, structured=structured, targets=labels,return_str_preds=True)
                    else:
                        pred, loss,acc = r.birds(model, imgs, targets=labels,return_str_preds=True)
                    val_loss += float(loss.detach().cpu().numpy())
                    val_acc+=acc
                val_loss/=len(test_dl)
                val_acc=(val_acc/len(test_dl))
                summary_writer.add_scalar('Val_loss', val_loss, step)

                print('Step %d, birds test loss: %f, Accuracy %f %%' % (step, val_loss,val_acc))
                log_str += 'Step %d, birds test loss: %f, Accuracy %f %%\n' % (step, val_loss,val_acc)

                model = model.train()

        elif task=='hmdb':
            if evaluating(log, eval_interval, i):
                if i == (start+1) and not eval_first:
                    continue
            # if (log and eval_interval is not None and i % eval_interval == 0):
                model = model.eval()
                val_loss = 0
                val_acc=0
                print('-' * 100)
                print('Evaluation step')
                for b in tqdm(val_dl):
                    vid,labels = b
                    if gpu_id >= 0:
                        vid = vid.cuda(device=gpu_id)
                        labels = labels.cuda(device=gpu_id)    
                    _, loss,acc = r.hmdb(model, vid,targets=labels, mode='val')
                    val_loss += float(loss.detach().cpu().numpy())
                    val_acc+=acc
                val_loss/=len(val_dl)
                val_acc=(val_acc/len(val_dl))
                summary_writer.add_scalar('Val_loss', val_loss, step)
                print('Step %d, HMDB validation loss: %f, Accuracy %f %%' % (step, val_loss,val_acc))
                print('-' * 100)
                model = model.train()
                continue
            vid,labels = next(DL)
            if gpu_id >= 0:
                vid = vid.cuda(device=gpu_id)
                labels = labels.cuda(device=gpu_id)    
            _, loss,acc = r.hmdb(model, vid,targets=labels,return_str_preds=True)
            loss.backward()
            loss=loss.detach()
            if log:
                summary_writer.add_scalar('Loss', loss, step)
            print('Step %d, HMDB Loss: %f, Accuracy:  %f %%' % (step, loss,acc))
            
        elif task == 'penn':
            if evaluating(log, eval_interval, i):
                if i == (start+1) and not eval_first:
                    continue
            # if (log and eval_interval is not None and i % eval_interval == 0):
                model = model.eval()
                val_loss=0
                val_acc=0
                print('-' * 100)
                print('Evaluation step')
                for b in tqdm(test_dl):
                    en = b['text']
                    targets = b['tokens']
                    pad_id=b['pad_id']
                    pad_mask=b['pad_mask']
                    if gpu_id>=0:
                        targets=targets.to(gpu_id)
                        pad_mask=pad_mask.to(gpu_id)
                    _,loss,acc = r.penn(model, en, target_pad_mask=pad_mask,
                                        pad_id=pad_id,targets=targets, mode='val',return_str_preds=True)
                    loss=loss.detach()
                    val_loss += float(loss.cpu().numpy())
                    val_acc+=acc
                val_loss/=len(val_dl)
                val_acc=(val_acc/len(val_dl))
                summary_writer.add_scalar('Val_loss', val_loss, step)
                print('Step %d, PENN validation loss: %f, Accuracy %f %%' % (step, val_loss,val_acc))
                print('-' * 100)
                model = model.train()
            batch = next(DL)
            en = batch['text']
            targets = batch['tokens']
            pad_id=batch['pad_id']
            pad_mask=batch['pad_mask']
            if gpu_id>=0:
                targets=targets.to(gpu_id)
                pad_mask=pad_mask.to(gpu_id)
            _, loss,acc = r.penn(model, en, pad_id=pad_id, targets=targets,target_pad_mask=pad_mask)
            loss.backward()
            loss=loss.detach()
            if log:
                summary_writer.add_scalar('Loss', loss, step)
            print('Step %d, PENN Loss: %f, Accuracy:  %f %%' % (step, loss,acc))
            
        # End Calculate loss
        if gpu_id>0:
            ensure_shared_grads(model, shared_model, gpu_id)

        optimizer.step()
        # Save model
        if (save_interval != None and (i+1) % save_interval == 0):
            # if task == 'vqa_struct':
            #     with open(model_save_path+'_%s_last_batch_questions.pkl'%step, 'wb') as f:
            #         pickle.dump(batch['ques'], f)

            if not save_best and not test:
                if step > save_interval:
                    os.rename(os.path.join(model_save_path,str(step-save_interval)),
                        os.path.join(model_save_path,str(step)))

                shared_model.save(model_save_path, step)
                optimizer.save(model_save_path, step)

            if save_best and ((i+1)//save_interval + 1) * save_interval >= train_steps:
                shared_model.save(move_out_path, 'last/0')
                optimizer.save(move_out_path, 'last/0')

            if task == 'vqa_struct' and (args.inject_at_encoder or args.inject_after_encoder) and args.spat_fusion_attn_type != 'none': 
                with open(model_save_path+'_spat_struct_enc_gate_tr_%s.dict'%i, 'wb') as f:
                    pickle.dump(spat_struct_enc_gate_tr, f)
            if task == 'vqa_struct' and (args.inject_at_encoder or args.inject_after_encoder) and args.temp_fusion_attn_type != 'none': 
                with open(model_save_path+'_temp_struct_enc_gate_tr_%s.dict'%i, 'wb') as f:
                    pickle.dump(temp_struct_enc_gate_tr, f)


            # with open(model_save_path + '.log', 'a') as f: #open(os.path.join(model_save_path, 'log.txt'), 'a') as f:
            #     print(log_str, file=f)

            try:
                with open(move_out_path + '.log', 'a') as f: #open(os.path.join(model_save_path, 'log.txt'), 'a') as f:
                    print(log_str, file=f)
                    log_str = ''
            except:
                pass

            # try:
            #     os.stat(model_save_path)
            # except:
            #     os.mkdir(model_save_path)
            # torch.save(best_model, os.path.join(model_save_path, 'best_val_model.pth'))
            # print('Best Model saved')


        sys.stdout.flush()

    if move_out_path is not None:
        with open(move_out_path + '.log', 'a') as f: #open(os.path.join(model_save_path, 'log.txt'), 'a') as f:
            print(log_str, file=f)
            
        # with open(model_save_path + '.log', 'a') as f:
        #     print(log_str, file=f)
        # with open(move_out_path + '_selected_ins_ids_by_seq.dict', 'wb') as f:
        #     pickle.dump(selected_ins_ids_by_seq, f)
        # with open(move_out_path + '_selected_ins_ids_all.dict', 'wb') as f:
        #     pickle.dump(selected_ins_ids_all, f)

        if not test and not save_best:
            copydir(os.path.join(model_save_path, str(best_iteration)), 
                move_out_path + '_' + str(best_iteration))

            ckpts = glob.glob(os.path.join(model_save_path, '*'))
            iters = [int(os.path.basename(c)) for c in ckpts]
            if len(iters) != 0:
                last = max(iters)

            copydir(os.path.join(model_save_path, str(last)), 
                move_out_path + '_' + str(last))
            # else:
            #     copydir(os.path.join(model_save_path, 'best/0'), 
            #         os.path.join(move_out_path + 'best/0'))

    # shutil.copy2(
    #     model_save_path + '.log',
    #     move_out_path + '.log')

def copydir(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copydir(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)
    
if __name__ == '__main__':
    # print('setting seeds:', int(args.torch_seed), int(args.random_seed), int(args.numpy_seed))
    # torch.manual_seed(int(args.torch_seed))#(47)
    # random.seed(int(args.random_seed))#(983)
    # np.random.seed(int(args.numpy_seed))#(1024)
    # torch.backends.cudnn.deterministic = True

    # if args.move_out_path and not os.path.exists(args.move_out_path):
    #     os.makedirs(args.move_out_path)

    mp.set_start_method('spawn',force=True)
    n_iters = int(args.n_iters)
    n_jobs = int(args.n_jobs)
    tasks=args.tasks
    batch_sizes=args.batch_sizes
    save_interval = int(int(args.save_interval) / n_jobs)
    eval_interval = int(int(args.eval_interval) / n_jobs)

    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

    if args.move_out_path and not os.path.exists(args.move_out_path):
        os.makedirs(args.move_out_path)

    if args.restore_last == True:
        ckpts = glob.glob(os.path.join(args.model_save_path, '*'))
        iters = [int(os.path.basename(c)) for c in ckpts if os.path.basename(c) != 'best']
        if len(iters) != 0:
            restore = max(iters)
        else:
            restore = -1
    else:
        restore = int(args.restore)
    tasks=tasks.split(',')
    tasks=[t.strip() for t in tasks]
    batch_sizes=batch_sizes.split(',')
    batch_sizes=[int(b.strip()) for b in batch_sizes]

    if len(tasks)!=len(batch_sizes):
        raise Exception('Number of tasks provided does not match the number of batch sizes provided.')

    n_gpus = int(args.n_gpus)
    n_tasks = len(tasks) * n_jobs

    if args.conf_type == 'default':
        config = defaultconf()
    elif args.conf_type == 'timeline':
        print('using timeline_conf')
        config = timeline_conf()
    elif args.conf_type == 'vqa_struct':
        print('using vqa_struct_config')
        config = vqa_struct_config()

        config[0]['temp_fusion_attn_type'] = args.temp_fusion_attn_type
        config[0]['spat_fusion_attn_type'] = args.spat_fusion_attn_type
        config[0]['inject_at_logits'] = args.inject_at_logits
        config[0]['inject_at_encoder'] = args.inject_at_encoder
        config[0]['inject_after_encoder'] = args.inject_after_encoder
        config[0]['convex_gate'] = args.convex_gate
        config[0]['pooling'] = args.pooling
        config[0]['no_logit_struct_peripheral'] = args.no_logit_struct_peripheral
        config[0]['logit_struct_periph_dim'] = args.logit_struct_periph_dim
        config[1]['struct_dropout'] = args.struct_periph_dropout
        config[1]['struct_temp_dropout'] = args.struct_temp_periph_dropout
        config[1]['struct_spat_dropout'] = args.struct_spat_periph_dropout

        if args.inject_at_decoder:
            config[0]['inject_at_decoder'] = True
            config[0]['use_s_decoder'] = True

        for periph_name in args.unfreeze:
            config[1]['unfreeze'][periph_name] = True

    elif args.conf_type == 'timeline_small':
        print('using timeline_conf_small')
        config = timeline_conf_small()
    
    # print(config)
    shared_model = omninet.OmniNet(gpu_id=0, config=config , peripherals_type=args.peripherals_type, unstructured_as_structured=args.unstructured_as_structured)

    print('num of parameters:', sum(p.numel() for p in shared_model.parameters() if p.requires_grad))

    eval_first = False
    if restore != -1:
        if args.save_best:
            shared_model.restore(args.move_out_path, 'last/0')
        else:
            shared_model.restore(args.move_out_path, restore)
        with open(args.move_out_path + '.log', 'r') as f:
            log = f.read()
        if len(re.findall('Step %s,'%(restore+1), log)) == 0:
            eval_first = True
    else:
        restore=0
        print_log(config, args.move_out_path + '.log')
        print_log(args, args.move_out_path + '.log')

    print('send omninet to gpu')
    shared_model=shared_model.to(0)
    print('sent omninet to gpu')
    shared_model.share_memory()
    counters = [Counter(restore) for i in range(len(tasks))]
    barrier = mp.Barrier(n_tasks)
    start = int(restore / n_jobs)
    # Declare training processes for multi-gpu hogwild training
    print('starting processes')
    processes = []
    for i in range(n_tasks):
        #If more than one GPU is used, use first GPU only for model sharing
        if n_gpus>1:
            gpu_id=i%n_gpus
        else:
            gpu_id=0
        process = mp.Process(target=train, args=(shared_model, tasks[i % len(tasks)], batch_sizes[i % len(tasks)],
                                                 int(n_iters / n_jobs),
                                                 gpu_id, start, restore, counters[i % len(tasks)], barrier,
                                                 args.n_workers,
                                                 (save_interval if i == 0 else None),
                                                 (eval_interval if i < len(tasks) else None),
                                                 eval_first,
                                                 (True if i < len(tasks) else False),
                                                 args.peripherals_type, args.conf_type,
                                                 args.model_save_path, args.move_out_path, args.save_best, args.testing,
                                                 args.with_val, args.norm_weights, args.scale_weights, args.sample_weights_fn, 
                                                 args.sample_idx_fn, args.rewarding, args.skip_seqs, 
                                                 args.overfitting_threshold, args.overfitting_start,
                                                 args.stop_overfitted_ins, args.detect_overfitting_with_tr_acc,
                                                 args.random_seq, args.notest, args.test, args.evaluate,
                                                 args.pick_sample_by_intsance_id,
                                                 args.pick_seq, args.use_sgd, args.prob_fn))
        process.start()
        processes.append(process)
    for p in processes:
        p.join()

