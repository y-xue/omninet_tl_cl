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
parser.add_argument('--torch_seed', default=47, help='torch manual_seed')
parser.add_argument('--random_seed', default=983, help='random seed')
parser.add_argument('--numpy_seed', default=1024, help='numpy seed')
parser.add_argument('--all_seed', default=1029, help='seed')
parser.add_argument('--save_best', action='store_true', help='True if only save the model on validation.')
parser.add_argument('--rewarding', action='store_true', help='True if evaluating model with reward function.')
parser.add_argument('--skip_seqs', default=None, nargs='+', help='skill seq in training')
parser.add_argument('--overfitting_threshold', default=1, type=float, help='threshold to determine if model is overfitting')
parser.add_argument('--overfitting_start', default=1000, type=float, help='iterations when to start handling overfitting')
parser.add_argument('--stop_overfitted_ins', action='store_true', help='True if stop training on instances that contains samples of overfitted sequence types.')
parser.add_argument('--detect_overfitting_with_tr_acc', action='store_true', help='True if determine overfitting by looking at training accuracy.')
parser.add_argument('--eval_first', action='store_true', help='True if evaluate model in the first iteration.')
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
parser.add_argument('--split', default='tr', type=str, help='Get saliency map on a particular split of data.')
parser.add_argument('--no_logit_struct_peripheral', action='store_true', help='True if no struct peripheral at logits.')


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
            config[1]['struct_dropout'] = args.struct_periph_dropout
            config[1]['struct_temp_dropout'] = args.struct_temp_periph_dropout
            config[1]['struct_spat_dropout'] = args.struct_spat_periph_dropout
            if args.inject_at_decoder:
                config[0]['inject_at_decoder'] = True
                config[0]['use_s_decoder'] = True
        
        if restore == 0:
            print_log(config, move_out_path + '.log')
        # print(config)
        model = omninet.OmniNet(gpu_id=gpu_id, config=config , peripherals_type=peripherals_type)
        model=model.cuda(gpu_id)
    else:
        #For GPU 0, use the shared model always
        model=shared_model

    structured_raw_grads_lst = []
    structured_temp_grads_lst = []
    structured_spat_grads_lst = []
    target_lst = []

    temp_cache_grads_dict = {'mean': [], 'max': [], 'min': [], 'std': []}
    spat_cache_grads_dict = {'mean': [], 'max': [], 'min': [], 'std': []}

    cache_grads_dict = {'temp': [], 'spat': []}

    if task == 'vqa' or task == 'vqa_struct':
        if task == 'vqa':
            DL,val_dl,test_dl = dl.vqa_batchgen(vqa_dir, coco_images, num_workers=n_workers, batch_size=batch_size, with_val=with_val, use_boxes=args.use_boxes, make_tr_dl_iter=False, drop_last_tr=False)
        else:
            DL,val_dl,test_dl = dl.vqa_batchgen(vqa_dir, coco_images, num_workers=n_workers, batch_size=batch_size, structured_path=structured_path, with_val=with_val, use_boxes=args.use_boxes, make_tr_dl_iter=False, drop_last_tr=False)
            
            if args.split == 'tr':
                dataloader = DL
            elif args.split == 'val':
                dataloader = val_dl
            else:
                dataloader = test_dl

            emb_loss_spat = None
            emb_loss_temp = None

        if use_sgd:
            print('SGD')
            optimizer = ScheduledOptim(
                SGD(
                    filter(lambda x: x.requires_grad, shared_model.parameters()),
                    lr=0.0001, momentum=0.9),
                512, 16000,restore,max_lr=0.0001,init_lr=args.init_lr)
        else:
            optimizer = ScheduledOptim(
                Adam(
                    filter(lambda x: x.requires_grad, shared_model.parameters()),
                    betas=(0.9, 0.98), eps=1e-09),
                512, 16000,restore,max_lr=0.0001,init_lr=args.init_lr)

    # if restore != 0:
    #     optimizer.restore(move_out_path, restore)

    # model=model.train()
    model = model.eval()

    best_val_acc = 0
    best_val_reward = 0
    if test:
        best_iteration = restore
        train_steps = start + 2
    else:
        best_iteration = save_interval
    log_str = ''

    i = 0
    for batch in dataloader:
        model.zero_grad()
        if barrier is not None:
            barrier.wait()
        if gpu_id > 0:
            with torch.cuda.device(gpu_id):
                model.load_state_dict(shared_model.state_dict())
                       
        # Calculate loss
        step = counter.increment()

        if task == 'vqa' or task == 'vqa_struct':
            # start_time = time.time()
            # # batch = next(DL)
            # end_time = time.time()
            # log_str += 'fetching next batch takes {:.2f}s\n'.format(end_time - start_time)
            # print('fetching next batch takes {:.2f}s'.format(end_time - start_time))
            
            start_time = time.time()
            if gpu_id >= 0:
                imgs = batch['img'].cuda(device=gpu_id)
                answers = batch['ans'].cuda(device=gpu_id)
            else:
                imgs = batch['img']
                answers = batch['ans']
            questions = batch['ques']
            end_time = time.time()
            # log_str += 'send next batch to gpu takes {:.2f}s\n'.format(end_time - start_time)
            # print('send next batch gpu takes {:.2f}s'.format(end_time - start_time))
            # print(questions[0], questions[-1])
            # print(imgs[0,0,5,:10], imgs[-1,0,5,-10:])
            # print(answers[0], answers[-1])

            start_time = time.time()
            if task == 'vqa_struct':
                structured = batch['struct'].cuda(device=gpu_id)
                structured.requires_grad=True

                structured_raw = structured.clone()
                structured_temp = structured.clone()
                structured_spat = structured.clone()

                structured_raw.register_hook(lambda grad: structured_raw_grads_lst.append(grad.detach().cpu().numpy()))
                structured_temp.register_hook(lambda grad: structured_temp_grads_lst.append(grad.detach().cpu().numpy()))
                structured_spat.register_hook(lambda grad: structured_spat_grads_lst.append(grad.detach().cpu().numpy()))

                target_lst.append(answers.detach().cpu().numpy().reshape(-1))

                bbox_mask = batch['bbox_mask']
                if bbox_mask is not None:
                    bbox_mask = bbox_mask.cuda(device=gpu_id)

                if args.fusion:
                    _, loss, acc, emb_loss_spat, emb_loss_temp, spat_struct_enc_gate, temp_struct_enc_gate = r.vqa_fusion_saliency(model, imgs, questions, structured_raw, structured_temp, structured_spat, targets=answers, bbox_mask=bbox_mask)
                
                    if emb_loss_spat is not None and emb_loss_temp is not None:
                        loss = loss + emb_loss_spat + emb_loss_temp
                    # emb_loss = r.vqa_struct_emb(model, imgs, questions, structured)
                    # total_loss = loss + emb_loss
                else:
                    _, loss, acc = r.vqa_saliency(model, imgs, questions, structured_raw, structured_temp, structured_spat, targets=answers, bbox_mask=bbox_mask, cache_grads_dict=cache_grads_dict)
            else:
                _, loss, acc = r.vqa_saliency(model, imgs, questions, targets=answers)
            end_time = time.time()
            log_str += 'forward pass takes {:.2f}s\n'.format(end_time - start_time)
            print('forward pass takes {:.2f}s'.format(end_time - start_time))

            # start_time = time.time()
            # emb_loss.backward(retain_graph=True)
            # end_time = time.time()
            # log_str += 'emb loss backward pass takes {:.2f}s\n'.format(end_time - start_time)
            # print('emb loss backward pass takes {:.2f}s'.format(end_time - start_time))

            # print(loss.detach())

            start_time = time.time()
            loss.backward()

            # structured_raw_grads = structured_raw.grad.data
            # structured_temp_grads = structured_temp.grad.data
            # structured_spat_grads = structured_spat.grad.data

            # print('structured_raw_grads:', structured_raw_grads_lst[-1].shape)
            # print(structured_raw_grads_lst[-1])
            # print('structured_temp_grads:', structured_temp_grads_lst[-1].shape)
            # print(structured_temp_grads_lst[-1])
            # print('structured_spat_grads:', structured_spat_grads_lst[-1].shape)
            # print(structured_spat_grads_lst[-1])


            # structured_raw_grads_lst.append(structured_raw_grads) 
            # structured_temp_grads_lst.append(structured_temp_grads) 
            # structured_spat_grads_lst.append(structured_spat_grads)

            if 'temp' in cache_grads_dict:
                # print('temp_cache_grads_mean:')
                # print(cache_grads_dict['temp'][0].mean(1))
                temp_cache_grads_dict['mean'].append(np.abs(cache_grads_dict['temp'][0]).mean(1))
            if 'spat' in cache_grads_dict:
                # print('spat_cache_grads_mean:')
                # print(cache_grads_dict['spat'][0].mean(1))
                spat_cache_grads_dict['mean'].append(np.abs(cache_grads_dict['spat'][0]).mean(1))


            cache_grads_dict['temp'] = []
            cache_grads_dict['spat'] = []

            end_time = time.time()
            log_str += 'backward pass takes {:.2f}s\n'.format(end_time - start_time)
            print('backward pass takes {:.2f}s'.format(end_time - start_time))

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

        # End Calculate loss
        if gpu_id>0:
            ensure_shared_grads(model, shared_model, gpu_id)

        optimizer.step()
        # Save model
        if (save_interval != None and (i+1) % save_interval == 0):
            if not save_best and not test:
                shared_model.save(model_save_path, step)
                optimizer.save(model_save_path, step)

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

        # if step >= train_steps + 1:
        #     break

        sys.stdout.flush()

    if move_out_path is not None:
        print('0')
        with open(move_out_path + '.log', 'a') as f: #open(os.path.join(model_save_path, 'log.txt'), 'a') as f:
            print(log_str, file=f)

        print('stacking')
        structured_raw_grads = np.vstack(structured_raw_grads_lst)
        structured_temp_grads = np.vstack(structured_temp_grads_lst)
        structured_spat_grads = np.vstack(structured_spat_grads_lst)
        
        temp_cache_grads_mean = np.vstack(temp_cache_grads_dict['mean'])
        spat_cache_grads_mean = np.vstack(spat_cache_grads_dict['mean'])

        all_targets = np.concatenate(target_lst)

        print('writing')
        with open(move_out_path+'_structured_raw_grads_%s.ndarray.pkl'%args.split, 'wb') as f:
            pickle.dump(structured_raw_grads, f)
        with open(move_out_path+'_structured_temp_grads_%s.ndarray.pkl'%args.split, 'wb') as f:
            pickle.dump(structured_temp_grads, f)
        with open(move_out_path+'_structured_spat_grads_%s.ndarray.pkl'%args.split, 'wb') as f:
            pickle.dump(structured_spat_grads, f)

        with open(move_out_path+'_temp_cache_grads_mean_%s.ndarray.pkl'%args.split, 'wb') as f:
            pickle.dump(temp_cache_grads_mean, f)

        with open(move_out_path+'_spat_cache_grads_mean_%s.ndarray.pkl'%args.split, 'wb') as f:
            pickle.dump(spat_cache_grads_mean, f)

        with open(move_out_path+'_target_lst_%s.ndarray.pkl'%args.split, 'wb') as f:
            pickle.dump(all_targets, f)

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
        iters = [int(os.path.basename(c)) for c in ckpts]
        if len(iters) != 0:
            restore = max(iters)
        else:
            restore = 0
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
        config[1]['struct_dropout'] = args.struct_periph_dropout
        config[1]['struct_temp_dropout'] = args.struct_temp_periph_dropout
        config[1]['struct_spat_dropout'] = args.struct_spat_periph_dropout
        if args.inject_at_decoder:
            config[0]['inject_at_decoder'] = True
            config[0]['use_s_decoder'] = True

    elif args.conf_type == 'timeline_small':
        print('using timeline_conf_small')
        config = timeline_conf_small()
    
    # print(config)
    shared_model = omninet.OmniNet(gpu_id=0, config=config , peripherals_type=args.peripherals_type)

    print('num of parameters:', sum(p.numel() for p in shared_model.parameters() if p.requires_grad))

    if restore != -1:
        shared_model.restore(args.model_save_path, restore)
    else:
        restore=0
        print_log(config, args.move_out_path + '.log')
        print_log(args, args.move_out_path + '.log')

    move_out_path = args.move_out_path + '_saliency_%s'%args.split
    model_save_path = args.model_save_path + '_saliency_%s'%args.split
    
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
                                                 args.eval_first,
                                                 (True if i < len(tasks) else False),
                                                 args.peripherals_type, args.conf_type,
                                                 model_save_path, move_out_path, args.save_best, args.testing,
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

