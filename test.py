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
import random
import sys
import pickle
from tqdm import tqdm
from libs.utils.train_util import *


# coco_images = 'data/coco/train_val'
# caption_dir = 'data/coco'
# vqa_dir = 'data/vqa'
# model_save_path = 'checkpoints'
# hmdb_data_dir='data/hmdb'
# hmdb_process_dir='data/hmdbprocess'
# penn_data_dir='data/penn'


# coco_images = '/data/gluster/brick1/yxue/research/omninet/data/coco/train_val'
# caption_dir = '/data/gluster/brick1/yxue/research/omninet/data/coco'
# vqa_dir = '/data/gluster/brick1/yxue/research/omninet/data/vqa'
# model_save_path = '/data/gluster/brick1/yxue/research/omninet/checkpoints/mm_vqa_labelFirst_test'
# hmdb_data_dir='/data/gluster/brick1/yxue/research/omninet/data/hmdb'
# hmdb_process_dir='/data/gluster/brick1/yxue/research/omninet/data/hmdbprocess'
# #hmdb_process_dir=''
# penn_data_dir='/data/gluster/brick1/yxue/research/omninet/data/penn'

coco_images = '/mnt-gluster/data/yxue/research/omninet_data/coco/train_val'
caption_dir = '/mnt-gluster/data/yxue/research/omninet_data/coco'
vqa_dir = '/mnt-gluster/data/yxue/research/omninet_data/vqa'
hmdb_data_dir='/mnt-gluster/data/yxue/research/omninet_data/hmdb'
hmdb_process_dir='/mnt-gluster/data/yxue/research/omninet_data/hmdbprocess'
#hmdb_process_dir=''
penn_data_dir='/mnt-gluster/data/yxue/research/omninet_data/penn'

# coco_images = '/data/gluster/brick1/yxue/research/omninet/data/coco/train_val'
# caption_dir = '/data/gluster/brick1/yxue/research/omninet/data/coco'
# vqa_dir = '/data/gluster/brick1/yxue/research/omninet/data/vqa'
# hmdb_data_dir='/data/gluster/brick1/yxue/research/omninet/data/hmdb'
# hmdb_process_dir='/data/gluster/brick1/yxue/research/omninet/data/hmdbprocess'
# #hmdb_process_dir=''
# penn_data_dir='/data/gluster/brick1/yxue/research/omninet/data/penn'
birds_dir='/home/yxue/research/allstate/data/CUB_200_2011' #'/mnt-gluster/data/yxue/research/allstate/data/CUB_200_2011'

mm_image_dir = '/mnt-gluster/data/yxue/research/allstate/data/cifar2'
mm_text_dir = '/mnt-gluster/data/yxue/research/allstate/data/aclImdb_new'
mm_video_dir = '/mnt-gluster/data/yxue/research/allstate/data/hmdb2'
mm_video_process_dir = '/mnt-gluster/data/yxue/research/allstate/data/hmdb2process_val'

# mm_video_dir = '/mnt-gluster/data/yxue/research/omninet_data/hmdb2'
# mm_video_process_dir = '/mnt-gluster/data/yxue/research/omninet_data/hmdb2process'
# mm_video_dir = '/mnt-gluster/data/yxue/research/allstate/data/hmdb2'
# mm_video_process_dir = '/mnt-gluster/data/yxue/research/allstate/data/hmdb2process'
mm_dir = '/mnt-gluster/data/yxue/research/allstate/data/synthetic_mm_cifar_imdb_hmdb'

# # model_save_path = '/home/yxue/research/allstate/code/omninet_struct/checkpoints/birds_struct' #'/mnt-gluster/data/yxue/research/allstate/omninet_struct/checkpoints/birds'
# model_save_path = '/mnt-gluster/data/yxue/research/allstate/omninet_struct/checkpoints/mm_ITIT_p.6.6_100000' #mm_vqa_labelFirst_seqMiniBatch_test'

def send_to_device(x, gpu_id):
    if x is not None:
        return x.cuda(device=gpu_id)
    return x

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def train(shared_model, task, batch_size, train_steps, gpu_id, start,  restore,  counter, barrier=None, save_interval=None,
          eval_interval=None, log=True, model_save_path='/out'):
    log_dir = 'logs/%s' % task
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if (log == True):
        summary_writer = SummaryWriter(log_dir)
    # Create local model
     
    torch.manual_seed(int(random.random() * 1000))
    if gpu_id>0:
        model = omninet.OmniNet(gpu_id=gpu_id)
        model=model.cuda(gpu_id)
    else:
        #For GPU 0, use the shared model always
        model=shared_model

    if task == 'caption':
        DL,val_dl = dl.coco_cap_batchgen(caption_dir=caption_dir, image_dir=coco_images,
                                  num_workers=0,
                                  batch_size=batch_size)
        
        optimizer = ScheduledOptim(
            Adam(
                filter(lambda x: x.requires_grad, shared_model.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            512, 16000,restore,init_lr=0.02)
    elif task == 'vqa':
        DL,val_dl = dl.vqa_batchgen(vqa_dir, coco_images, num_workers=0, batch_size=batch_size)
        optimizer = ScheduledOptim(
            Adam(
                filter(lambda x: x.requires_grad, shared_model.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            512, 16000,restore,max_lr=0.0001,init_lr=0.02)
    # elif task == 'vqa_struct':
    #     DL,val_dl = dl.vqa_struct_batchgen(vqa_dir, coco_images, num_workers=0, batch_size=batch_size)
    #     optimizer = ScheduledOptim(
    #         Adam(
    #             filter(lambda x: x.requires_grad, shared_model.parameters()),
    #             betas=(0.9, 0.98), eps=1e-09),
    #         512, 16000,restore,max_lr=0.0001,init_lr=0.02)
    elif task == 'birds' or task == 'birds_struct':
        DL,val_dl = dl.birds_batchgen(birds_dir, num_workers=0, batch_size=batch_size)
        optimizer = ScheduledOptim(
            Adam(
                filter(lambda x: x.requires_grad, shared_model.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            512, 16000,restore,max_lr=0.0001,init_lr=0.02)
    elif task == 'mm_ITV':
        seq_lst = ['0-1-0', '0-2-0', '1-0-0', '1-1-0', '1-2-0', '2-1-0', 
                    '0-1-1', '1-2-1', '2-2-0', '2-2-1', '0-2-1', '1-1-1', 
                    '0-0-1', '2-0-0', '2-1-1', '1-0-1']
        full_seq = 'ITTIV'
        with open(os.path.join(mm_dir, 'sample_idx_ITTIV_pT.9_35k-15k-15k'), 'rb') as f:
            sample_idx = pickle.load(f)

        dl_lst, val_dl_lst, test_dl_lst = dl.mm_batchgen(full_seq, seq_lst, mm_image_dir, mm_text_dir, mm_video_dir, mm_video_process_dir, sample_idx, num_workers=0, batch_size=batch_size)
        tr_seq_sizes = [len(dataset) for dataset in dl_lst]
        # [11368, 9275, 2542, 2893, 3693, 79, 126, 248, 382, 28, 555, 54, 8, 4, 5, 2] for batch_size 5

        # val_seq_sizes = [len(dataset) for dataset in val_dl_lst]
        # [4864, 3989, 1099, 1240, 1604, 32, 47, 107, 159, 10, 238, 25, 3, 3, 3, 1] for batch_size 5

        # test_seq_sizes = [len(dataset) for dataset in test_dl_lst]
        # [4854, 3948, 1110, 1258, 1616, 38, 56, 103, 166, 11, 240, 25, 3, 1, 2, 2] for batch_size 5

        # 0.4s/it for batch_size 8

        p = np.array(tr_seq_sizes)/sum(tr_seq_sizes)
        DLS = [iter(cycle(tr_dl)) for tr_dl in dl_lst]

        # DLS = iter(cycle(dl_lst))
        # seq_it = iter(cycle(seq_lst))

        optimizer = ScheduledOptim(
            Adam(
                filter(lambda x: x.requires_grad, shared_model.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            512, 16000,restore,max_lr=0.0001,init_lr=0.02)
    elif task == 'mm_IT':
        seq_lst = ['0-1', '1-0', '1-1', '2-1', '1-2', '2-0', '2-2', '0-2']
        full_seq = 'ITIT'
        with open(os.path.join(mm_dir, 'sample_idx_ITIT_p.6.6_100000'), 'rb') as f:
            sample_idx = pickle.load(f)

        dl_lst, val_dl_lst = dl.mm_batchgen(full_seq, seq_lst, mm_image_dir, mm_text_dir, mm_video_dir, mm_video_process_dir, sample_idx, num_workers=0, batch_size=batch_size)

        DLS = iter(cycle(dl_lst))
        seq_it = iter(cycle(seq_lst))

        optimizer = ScheduledOptim(
            Adam(
                filter(lambda x: x.requires_grad, shared_model.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            512, 16000,restore,max_lr=0.0001,init_lr=0.02)
    elif task == 'mm_vqa':
        DL0,DL1,DL2,DL3,val_dl = dl.mm_vqa_batchgen(vqa_dir, coco_images, num_workers=0, batch_size=batch_size)
        DLS = iter(cycle([DL0,DL1,DL2,DL3]))
        # DL, val_dl = dl.mm_vqa_batchgen(vqa_dir, coco_images, num_workers=0, batch_size=batch_size)

        # DL,val_dl = dl.mm_batchgen(mm_dir, num_workers=0, batch_size=batch_size)
        optimizer = ScheduledOptim(
            Adam(
                filter(lambda x: x.requires_grad, shared_model.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            512, 16000,restore,max_lr=0.0001,init_lr=0.02)
    
    elif task == 'hmdb':
        DL,val_dl=dl.hmdb_batchgen(hmdb_data_dir,hmdb_process_dir,num_workers=0,batch_size=batch_size,
                                   test_batch_size=int(batch_size/4),
                                   clip_len=16)
        optimizer = ScheduledOptim(
            Adam(
                filter(lambda x: x.requires_grad, shared_model.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            512, 16000,restore,max_lr=0.0001,init_lr=0.02)
    elif task == 'penn':
        DL,val_dl,test_dl=dl.penn_dataloader(penn_data_dir,batch_size=batch_size,
                                             test_batch_size=int(batch_size/2),num_workers=4,vocab_file='conf/penn_vocab.json')
        optimizer = ScheduledOptim(
            Adam(
                filter(lambda x: x.requires_grad, shared_model.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            512, 16000,restore,init_lr=0.02)
        
    model=model.train()

    best_val_acc = 0
    best_iteration = 0

    for i in range(start, train_steps):
        model.zero_grad()
        if barrier is not None:
            barrier.wait()
        if gpu_id > 0:
            with torch.cuda.device(gpu_id):
                model.load_state_dict(shared_model.state_dict())
                       
        # Calculate loss
        step = counter.increment()

        if task == 'mm_ITV':
            best_iteration = 1500
            # pretrained_dict=torch.load(os.path.join(model_save_path, 'best_val_model.pth'))
            # model_dict=shared_model.state_dict()
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if
            #                (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
            # model.load_state_dict(pretrained_dict,strict=False)
            # print(best_iteration)
            shared_model.restore(model_save_path, best_iteration)
            model = shared_model
            model = model.eval()
            print('-' * 100)
            print('Test model')
            total = 0
            total_correct = 0
            for seq, val_dl in zip(seq_lst[9:], test_dl_lst[9:]):
                print(seq)
                val_loss = 0
                loss_total = 0
                val_correct = 0
                val_total = 0
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

                summary_writer.add_scalar('Test_loss_%s'%seq, val_loss, step)
                print('Step %d, %s, mm test loss: %f, Accuracy %f %%' % (step, seq, val_loss,val_acc))
            print('-' * 100)
            total_val_acc = (total_correct/total)*100
            print('Step %d, mm total test Accuracy %f %%'%(step, total_val_acc))
            model = model.train()
            
        # End Calculate loss
        if gpu_id>0:
            ensure_shared_grads(model, shared_model, gpu_id)
        optimizer.step()
        # Save model
        if (save_interval != None and (i+1) % save_interval == 0):
            shared_model.save(model_save_path, step)

            # try:
            #     os.stat(model_save_path)
            # except:
            #     os.mkdir(model_save_path)
            # torch.save(best_model, os.path.join(model_save_path, 'best_val_model.pth'))
            # print('Best Model saved')


        sys.stdout.flush()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OmniNet training script.')
    parser.add_argument('n_iters', help='Number of iterations to train.')
    parser.add_argument('tasks', help='List of tasks seperated by comma.')
    parser.add_argument('batch_sizes', help='List of batch size for each task seperated by comma')
    parser.add_argument('--n_jobs', default=1, help='Number of asynchronous jobs to run for each task.')
    parser.add_argument('--n_gpus', default=1, help='Number of GPUs to use')
    parser.add_argument('--save_interval', default=100, help='Number of iterations after which to save the model.')
    parser.add_argument('--restore', default=-1, help='Step from which to restore model training')
    parser.add_argument('--restore_last', help='Restore the latest version of the model.', action='store_true')
    parser.add_argument('--eval_interval', help='Interval after which to evaluate on the test/val set.', default=1000)
    parser.add_argument('--model_save_path', default='/out/test', help='path to save the model.')

    args = parser.parse_args()
    torch.manual_seed(47)
    random.seed(983)
    np.random.seed(1024)
    torch.backends.cudnn.deterministic = True
    mp.set_start_method('spawn',force=True)
    n_iters = int(args.n_iters)
    n_jobs = int(args.n_jobs)
    tasks=args.tasks
    batch_sizes=args.batch_sizes
    save_interval = int(int(args.save_interval) / n_jobs)
    eval_interval = int(int(args.eval_interval) / n_jobs)

    if args.restore_last == True:
        ckpts = glob.glob(os.path.join(model_save_path, '*'))
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

    shared_model = omninet.OmniNet(gpu_id=0)
    if restore != -1:
        shared_model.restore(model_save_path, restore)
    else:
        restore=0
        
    shared_model=shared_model.to(0)
    shared_model.share_memory()
    counters = [Counter(restore) for i in range(len(tasks))]
    barrier = mp.Barrier(n_tasks)
    start = int(restore / n_jobs)
    # Declare training processes for multi-gpu hogwild training
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
                                                 (save_interval if i == 0 else None),
                                                 (eval_interval if i < len(tasks) else None),
                                                 (True if i < len(tasks) else False),
                                                 args.model_save_path))
        process.start()
        processes.append(process)
    for p in processes:
        p.join()


