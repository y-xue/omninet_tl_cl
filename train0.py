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
import random
import sys
import pickle
from tqdm import tqdm

import json
import shutil

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
parser.add_argument('--temp_fusion_attn_type', default='default', type=str, help='Temporal attention type in fusion.')
parser.add_argument('--spat_fusion_attn_type', default='default', type=str, help='Spatial attention type in fusion.')
parser.add_argument('--convex_gate', action='store_true', help='True if use convex gate.')
parser.add_argument('--use_sgd', action='store_true', help='True if use SGD.')
parser.add_argument('--prob_fn', default=None, type=str, help='file name of predefined sampling probablities by sequences')
parser.add_argument('--init_lr', default=0.02, type=float, help='init_lr')

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

if __name__ == '__main__':
    

    step = 0
    total_val_acc = 100*np.random.random()
    total_reward = np.random.uniform(0,5)
    
    if args.test:
        log_str = 'Step %d, mm total test Accuracy %f %%, reward: %f\n'%(step, total_val_acc, total_reward)
    else:
        log_str = 'Step %d, mm total validation Accuracy %f %%, reward: %f\n'%(step, total_val_acc, total_reward)

    with open(args.move_out_path + '.log', 'a') as f: #open(os.path.join(model_save_path, 'log.txt'), 'a') as f:
        print(log_str, file=f)
