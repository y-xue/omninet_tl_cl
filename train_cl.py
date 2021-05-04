#
# Copyright 2019 Subhojeet Pramanik, Aman Husain, Priyanka Agrawal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	 http://www.apache.org/licenses/LICENSE-2.0
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
parser.add_argument('--sample_idx_fn', default='sample_idx_ITTIV_pT.9_35k-15k-15k', help='file name of sample_idx.')
parser.add_argument('--peripherals_type', default='default', help='Choose peripherals types')
parser.add_argument('--conf_type', default='default', help='Choose confurigation types')
parser.add_argument('--torch_seed', default=47, help='torch manual_seed')
parser.add_argument('--data_seed', default=615, type=int, help='numpy seed')
parser.add_argument('--all_seed', default=1029, help='seed')
parser.add_argument('--save_best', action='store_true', help='True if only save the model on validation.')
parser.add_argument('--rewarding', action='store_true', help='True if evaluating model with reward function.')
parser.add_argument('--eval_first', action='store_true', help='True if evaluate model in the first iteration.')
parser.add_argument('--data_path', default='/files/yxue/research/allstate/data', help='data path')
parser.add_argument('--random_seq', action='store_true', help='Deprecated. True if randomly pick a seq to train in each iteration.')
parser.add_argument('--testing', action='store_true', help='True if training with validation set for a few epochs for testing.')
parser.add_argument('--notest', action='store_true', help='True if no test model.')
parser.add_argument('--test', action='store_true', help='True if only test model.')
parser.add_argument('--test_on_val', action='store_true', help='True if only test model on validation set.')
# parser.add_argument('--evaluate', action='store_true', help='True if only eval model.')
parser.add_argument('--use_sgd', action='store_true', help='True if use SGD.')
parser.add_argument('--init_lr', default=0.02, type=float, help='init_lr')
parser.add_argument('--n_warmup_steps', default=3000, type=float, help='n_warmup_steps for ScheduledOptim')
# parser.add_argument('--ewc_lambda', default=0.4, type=float, help='hyperparameter of EWC')
parser.add_argument('--ewc_lambda', default=[0.4], nargs='+', type=float, help='hyperparameter of EWC')
parser.add_argument('--scale_ewc_lambda', default=[1], nargs='+', type=int, help='scale ewc_lambda')
parser.add_argument('--full_seq', default='ITTIV', help='order of modalities')

parser.add_argument('--restore_cl', default=-1, help='Step from which to restore model training')
# parser.add_argument('--restore_cl_last', help='Restore the latest version of the model.', action='store_true')

parser.add_argument('--converge_window', default=10, type=int, help='Window size for convergence test')
parser.add_argument('--safe_window', default=30, type=int, help='Never stop training in safe_window')
parser.add_argument('--restore_opt', action='store_true', help='True if restore optimizer for each task.')
parser.add_argument('--restore_opt_lr', action='store_true', help='True if restore learning rate for each task.')

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
# structured_path = os.path.join(data_path, 'vqa', args.structured_folder)
# hmdb_data_dir = os.path.join(data_path, 'hmdb')
hmdb_data_dir = data_path
hmdb_process_dir = os.path.join(data_path, 'hmdbprocess')
# penn_data_dir = os.path.join(data_path, 'penn')

mm_image_dir = data_path
mm_text_dir = os.path.join(data_path, 'aclImdb_new')
mm_video_dir = data_path
mm_video_process_dir = os.path.join(data_path, 'hmdbprocess')

mm_dir = os.path.join(data_path, 'synthetic_mm_cifar_imdb_hmdb')

if args.restore_cl != -1:
	with open(args.model_save_path + '/fisher_dict.pkl', 'rb') as f:
		fisher_dict = pickle.load(f)
	with open(args.model_save_path + '/optpar_dict.pkl', 'rb') as f:
		optpar_dict = pickle.load(f)

	with open(args.model_save_path + '/current_iterations.pkl', 'rb') as f:
		current_iterations = pickle.load(f)
else:
	fisher_dict = {}
	optpar_dict = {}
	current_iterations = {}

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
		  save_best=False, testing=False, sample_idx_fn=None, seq_lst=None, rewarding=False,
		  random_seq=False, notest=False, test=False, test_on_val=False,
		  use_sgd=False,
		  task_id=0, ewc_lambda=0, full_seq='ITTIV',counting_reward_dict=None,pass_idx=None):
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

		if restore == 0:
			print_log(config, model_save_path + '.log')
		# print(config)
		model = omninet.OmniNet(gpu_id=gpu_id, config=config , peripherals_type=peripherals_type)
		model=model.cuda(gpu_id)
	else:
		#For GPU 0, use the shared model always
		model=shared_model

	log_str = ''
	current_modality = args.full_seq[task_id % len(args.full_seq)]
	if current_modality == 'V':
		safe_window = 10
	else:
		safe_window = args.safe_window

	if task == 'mm_ITV_CL':
		with open(os.path.join(mm_dir, sample_idx_fn), 'rb') as f:
			sample_idx = pickle.load(f)

		seq_idx = dict(zip(seq_lst, range(len(seq_lst))))
		
		predefined_sample_weights = dict(zip([str(x) for x in range(1,6)], [1.]*5))
		print(predefined_sample_weights)

		print('creating data loaders')
		dl_lst, val_dl_lst, test_dl_lst = dl.mm_batchgen(full_seq, seq_lst, 
			mm_image_dir, mm_text_dir, mm_video_dir, mm_video_process_dir,
			predefined_sample_weights, sample_idx, 
			num_workers=n_workers, batch_size=batch_size, seq_count=False, 
			reset_seed=pass_idx, data_seed=args.data_seed)

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

		DLS = [iter(cycle(tr_dl)) for tr_dl in dl_lst]
		dl_ids = iter(cycle(range(len(dl_lst))))

		optimizer = ScheduledOptim(
			Adam(
				filter(lambda x: x.requires_grad, shared_model.parameters()),
				betas=(0.9, 0.98), eps=1e-09),
			512, args.n_warmup_steps,0,max_lr=0.0001,init_lr=args.init_lr)

	if current_modality in current_iterations and (args.restore_opt or args.restore_opt_lr):
		optimizer.set_current_step(current_iterations[current_modality])
		log_str += 'restore optimizer iteration: %s\n'%(current_iterations[current_modality])

		if args.restore_opt:
			optimizer.restore(args.model_save_path, '%s/0'%current_modality)
			log_str += 'restore optimizer from %s/%s/0\n'%(args.model_save_path, current_modality)
		
	else:
		current_iterations[current_modality] = 0

	model=model.train()

	# best_val_acc = 0
	# best_val_reward = 0
	best_val_score = 0

	curr_max_val_score = 0
	prev_max_val_score = 0

	validation_score_history = []

	if test:
		best_iteration = restore
		train_steps = start + 2
	else:
		best_iteration = save_interval

	for i in range(start+1, train_steps):
		model.zero_grad()
		if barrier is not None:
			barrier.wait()
		if gpu_id > 0:
			with torch.cuda.device(gpu_id):
				model.load_state_dict(shared_model.state_dict())
					   
		# Calculate loss
		step = counter.increment()

		if task == 'mm_ITV_CL':
			if not notest and i + 1 >= train_steps:
				# pretrained_dict=torch.load(os.path.join(model_save_path, 'best_val_model.pth'))
				# model_dict=shared_model.state_dict()
				# pretrained_dict = {k: v for k, v in pretrained_dict.items() if
				#				(k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
				# model.load_state_dict(pretrained_dict,strict=False)
				# print(best_iteration)
				if save_best:
					shared_model.restore(model_save_path, 'best/0')
					# optimizer.restore(model_save_path, 'best/0')
				else:
					shared_model.restore(model_save_path, best_iteration)
					# optimizer.restore(model_save_path, best_iteration)

				log_str += 'Restored existing model with iterations: %d\n' % (best_iteration)
				model = shared_model
				model = model.eval()

				print('-' * 100)
				if test_on_val:
					test_dl_lst = val_dl_lst
					print('Test model on val')
					log_str += '-' * 100 + '\nTest model on val\n'
				else:
					print('Test model')
					log_str += '-' * 100 + '\nTest model\n'

				total = 0
				total_correct = 0
				total_reward = 0
				total_reward_t = 0

				for seq, val_dl in zip(seq_lst, test_dl_lst):
					print(seq)
					log_str += '{}\n'.format(seq)
					val_loss = 0
					loss_total = 0
					val_correct = 0
					val_total = 0
					val_reward = 0
					val_reward_t = 0
					# j=0
					for b in val_dl: #tqdm(val_dl):
						# if j > 16:
						# 	 break
						# j += 1
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
						val_reward = reward_ITTIV(seq, val_acc/100, count=False)
						total_reward += val_reward

						val_reward_t = reward_ITTIV_t(seq, val_acc/100, count=False)
						total_reward_t += val_reward_t

					if counting_reward_dict is not None:
						seqc = '%s-%s-%s'%(seq.count('I'), seq.count('T'), seq.count('V'))
						if seqc in counting_reward_dict:
							counting_reward_dict[seqc][0] += val_correct
							counting_reward_dict[seqc][1] += val_total
						else:
							counting_reward_dict[seqc] = [val_correct,val_total,0]
						
					summary_writer.add_scalar('Test_loss_%s'%seq, val_loss, step)
					print('Step %d, %s, mm test loss: %f, Accuracy %f %%, reward: %f, reward_t: %f (%d/%d)' % (step, seq, val_loss, val_acc, val_reward, val_reward_t, val_correct, val_total))
					log_str += 'Step %d, %s, mm test loss: %f, Accuracy %f %%, reward: %f, reward_t: %f (%d/%d)\n' % (step, seq, val_loss, val_acc, val_reward, val_reward_t, val_correct, val_total)

				if counting_reward_dict is not None:
					print('-' * 100)
					log_str += '-' * 100 + '\n'
					for seqc in counting_reward_dict:
						n_correct, n_total, _ = counting_reward_dict[seqc]
						acc = n_correct/n_total
						counting_reward_dict[seqc][2] = reward_ITTIV(seqc, acc, count=True)
						print('Step %d, %s, mm test Accuracy %f %%, reward: %f (%d/%d)' % (step, seqc,acc*100,counting_reward_dict[seqc][2],n_correct,n_total))
						log_str += 'Step %d, %s, mm test Accuracy %f %%, reward: %f (%d/%d)\n' % (step, seqc,acc*100,counting_reward_dict[seqc][2],n_correct,n_total)

				print('-' * 100)
				log_str += '-' * 100 + '\n'
				total_val_acc = (total_correct/total)*100
				print('Step %d, mm total test Accuracy %f %%, reward: %f, reward_t: %f (%d/%d)'%(step, total_val_acc, total_reward, total_reward_t, total_correct, total))
				log_str += 'Step %d, mm total test Accuracy %f %%, reward: %f, reward_t: %f (%d/%d)\n'%(step, total_val_acc, total_reward, total_reward_t, total_correct, total)

				if counting_reward_dict is not None:
					total_counting_reward = sum([v[2] for _,v in counting_reward_dict.items()])
					print('Step %d, mm total test reward: %f'%(step, total_counting_reward))
					log_str += 'Step %d, mm total test reward: %f\n'%(step, total_counting_reward)

			if not test and evaluating(log, eval_interval, i):
				if i == (start+1) and not eval_first:
					continue

				model = model.eval()
				print('-' * 100)
				print('Evaluation step')
				log_str += '-' * 100 + '\nEvaluation step\n'

				total = 0
				total_correct = 0
				total_reward = 0
				total_reward_t = 0

				val_counting_reward_dict = {}

				for seq, val_dl in zip(seq_lst, val_dl_lst):
					val_loss = 0
					loss_total = 0
					val_correct = 0
					val_total = 0
					val_reward = 0
					val_reward_t = 0
					
					val_dl_size = len(val_dl)
					
					print(seq, 'val size:', val_dl_size)
					log_str += '{} val size: {}\n'.format(seq, val_dl_size)
					start_time = time.time()
					
					# j = 0
					for b in val_dl:
						# if j > 16:
						# 	break
						# j += 1
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
						val_reward = reward_ITTIV(seq, val_acc, count=False)
						total_reward += val_reward

						val_reward_t = reward_ITTIV_t(seq, val_acc, count=False)
						total_reward_t += val_reward_t

					seqc = '%s-%s-%s'%(seq.count('I'), seq.count('T'), seq.count('V'))
					if seqc in val_counting_reward_dict:
						val_counting_reward_dict[seqc][0] += val_correct
						val_counting_reward_dict[seqc][1] += val_total
					else:
						val_counting_reward_dict[seqc] = [val_correct,val_total,0]
						
					summary_writer.add_scalar('Val_loss_%s'%seq, val_loss, step)
					print('Step %d, %s, mm validation loss: %f, Accuracy %f %%, reward: %f, reward_t: %f (%d/%d)' % (step, seq, val_loss,val_acc*100,val_reward,val_reward_t,val_correct,val_total))
					log_str += 'Step %d, %s, mm validation loss: %f, Accuracy %f %%, reward: %f, reward_t: %f (%d/%d)\n' % (step, seq, val_loss,val_acc*100,val_reward,val_reward_t,val_correct,val_total)
					end_time = time.time()
					print('Step {}, {}, validation takes {:.2f}s\n'.format(step, seq, end_time - start_time))
					log_str += 'Step {}, {}, validation takes {:.2f}s\n'.format(step, seq, end_time - start_time)

				print('-' * 100)
				log_str += '-' * 100 + '\n'
				for seqc in val_counting_reward_dict:
					n_correct, n_total, _ = val_counting_reward_dict[seqc]
					acc = n_correct/n_total
					val_counting_reward_dict[seqc][2] = reward_ITTIV(seqc, acc, count=True)
					print('Step %d, %s, mm validation Accuracy %f %%, reward: %f (%d/%d)' % (step, seqc,acc*100,val_counting_reward_dict[seqc][2],n_correct, n_total))
					log_str += 'Step %d, %s, mm validation Accuracy %f %%, reward: %f (%d/%d)\n' % (step, seqc,acc*100,val_counting_reward_dict[seqc][2],n_correct, n_total)


				print('-' * 100)
				log_str += '-' * 100 + '\n'

				total_val_acc = (total_correct/total)*100
				total_counting_reward = sum([v[2] for _,v in val_counting_reward_dict.items()])

				if rewarding:
					total_val_score = total_counting_reward
				else:
					total_val_score = total_val_acc

				validation_score_history.append(total_val_score)

				if total_val_score > best_val_score:
					best_val_score = total_val_score
					best_iteration = step-1
					print(best_iteration)
					log_str += 'best_iteration:{}\n'.format(best_iteration)

					if save_best:
						shared_model.save(model_save_path, 'best/0')

						if args.restore_opt:
							optimizer.save(args.model_save_path, '%s/0'%current_modality)

				# # if rewarding and total_reward > best_val_reward:
				# if rewarding and total_counting_reward > best_val_reward:
				# 	# best_val_reward = total_reward
				# 	best_val_reward = total_counting_reward
				# 	best_iteration = step-1
				# 	print(best_iteration)
				# 	log_str += 'best_iteration:{}\n'.format(best_iteration)

				# 	if save_best:
				# 		shared_model.save(model_save_path, 'best/0')
				# 		optimizer.save(model_save_path, 'best/0')

				# if not rewarding and total_val_acc > best_val_acc:
				# 	best_val_acc = total_val_acc
				# 	best_iteration = step-1
				# 	print(best_iteration)
				# 	log_str += 'best_iteration:{}\n'.format(best_iteration)

				# 	if save_best:
				# 		shared_model.save(model_save_path, 'best/0')
				# 		optimizer.save(model_save_path, 'best/0')
				# 	# best_model = shared_model.state_dict()
				# 	# shared_model.save(model_save_path, step, best_model=True)

				print('Step %d, mm total validation Accuracy %f %%, reward: %f, reward_t: %f'%(step, total_val_acc, total_reward, total_reward_t))
				log_str += 'Step %d, mm total validation Accuracy %f %%, reward: %f, reward_t: %f\n'%(step, total_val_acc, total_reward, total_reward_t)
				print('Step %d, mm total validation reward: %f'%(step, total_counting_reward))
				log_str += 'Step %d, mm total validation reward: %f\n'%(step, total_counting_reward)

				with open(model_save_path + '.log', 'a') as f: #open(os.path.join(model_save_path, 'log.txt'), 'a') as f:
					print(log_str, file=f)
					log_str = ''

				curr_max_val_score = max(curr_max_val_score, total_val_score)
			
				eval_step = i // eval_interval

				if eval_step >= args.converge_window:
					prev_max_val_score = max(prev_max_val_score, validation_score_history[int(eval_step-args.converge_window)])

					if curr_max_val_score <= prev_max_val_score and eval_step >= args.safe_window:
						break

				model = model.train()
				continue

			# train
			if not test:
				next_dl_id = np.random.choice(range(len(DLS)))
				DL = DLS[next_dl_id]
				seq = seq_lst[next_dl_id]
				batch = next(DL)
					
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
				structured=None
						
				_, loss, acc, _, _ = r.mm(model, imgs, text, videos, structured, targets=labels, sample_weights=sample_weights)

				fisher_loss_dict = {}
				if ewc_lambda > 0:
					if task_id < len(args.full_seq):
						prev_task_idx_lst = [ti for ti in range(task_id)]
					else:
						prev_task_idx_lst = [ti % len(args.full_seq) for ti in range(task_id-len(args.full_seq)+1, task_id)]

					for prev_task_idx in prev_task_idx_lst:
						fisher_loss_dict[prev_task_idx] = {'loss': 0, 'ewc_lambda': ewc_lambda}
						for name, param in model.named_parameters():
							if name not in fisher_dict[prev_task_idx]:
								continue
							fisher = fisher_dict[prev_task_idx][name]
							optpar = optpar_dict[prev_task_idx][name]
							
							fisher_loss = (fisher * (optpar - param).pow(2)).sum()
							loss += fisher_loss * ewc_lambda

							fisher_loss_dict[prev_task_idx]['loss'] += fisher_loss.detach()

					# if task_id < len(args.full_seq):
					# 	for ti in range(task_id):
					# 		for name, param in model.named_parameters():
					# 			if name not in fisher_dict[ti]:
					# 				continue
					# 			fisher = fisher_dict[ti][name]
					# 			optpar = optpar_dict[ti][name]
					# 			loss += (fisher * (optpar - param).pow(2)).sum() * ewc_lambda
					# else:
					# 	for ti in range(task_id-len(args.full_seq)+1, task_id):
					# 		tj = ti % len(args.full_seq)
					# 		for name, param in model.named_parameters():
					# 			if name not in fisher_dict[tj]:
					# 				continue
					# 			fisher = fisher_dict[tj][name]
					# 			optpar = optpar_dict[tj][name]
					# 			loss += (fisher * (optpar - param).pow(2)).sum() * ewc_lambda

				loss.backward()
				loss=loss.detach()
				if log:
					summary_writer.add_scalar('Loss_%s'%seq, loss, step)

				fisher_loss_str = ''
				for ti in range(len(args.full_seq)):
					if ti in fisher_loss_dict:
						fisher_loss_str += ' t%s: %f %s'%(ti, fisher_loss_dict[ti]['loss'], fisher_loss_dict[ti]['ewc_lambda'])
					else:
						fisher_loss_str += ' t%s: 0 0'%(ti)

				print('Step %d, %s, mm Loss: %f, Accuracy:  %f %%, fisher Loss:%s' % (step, seq, loss, acc, fisher_loss_str))
				log_str += 'Step %d, %s, mm Loss: %f, Accuracy:  %f %%, fisher Loss:%s\n' % (step, seq, loss, acc, fisher_loss_str)

		# End Calculate loss
		if gpu_id>0:
			ensure_shared_grads(model, shared_model, gpu_id)

		optimizer.step()
		# Save model
		if (save_interval != None and (i+1) % save_interval == 0):
			if not save_best and not test:
				shared_model.save(model_save_path, step)
				if args.restore_opt:
					optimizer.save(model_save_path, step)

			try:
				with open(model_save_path + '.log', 'a') as f: #open(os.path.join(model_save_path, 'log.txt'), 'a') as f:
					print(log_str, file=f)
					log_str = ''
			except:
				pass

		sys.stdout.flush()

	current_iterations[current_modality] += best_iteration

	if model_save_path is not None:
		with open(model_save_path + '.log', 'a') as f: #open(os.path.join(model_save_path, 'log.txt'), 'a') as f:
			print(log_str, file=f)

		if not test and not save_best:
			copydir(os.path.join(model_save_path, str(best_iteration)), 
				model_save_path + '_' + str(best_iteration))

			ckpts = glob.glob(os.path.join(model_save_path, '*'))
			iters = [int(os.path.basename(c)) for c in ckpts]
			if len(iters) != 0:
				last = max(iters)

			copydir(os.path.join(model_save_path, str(last)), 
				model_save_path + '_' + str(last))

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
	
def gen_seq_type_helper(full_seq, tasks):
	if len(tasks) == 0:
		seq = full_seq[0]+'?'*(len(full_seq)-1)
		# print([seq])
		return [seq]
	
	new_tasks = []
	k = len(tasks)
	m = full_seq[k]
	for i in range(k):
		for seq in tasks[i]:
			# print(seq[:k]+m+seq[(k+1):])
			new_tasks.append(seq[:k]+m+seq[(k+1):])
	new_tasks.append('?'*k+m+'?'*(len(full_seq)-k-1))
	# print(new_tasks)
	return new_tasks

def gen_seq_type(full_seq):
	tasks = []
	for i in range(len(full_seq)):
		# print('task:', i)
		tasks.append(gen_seq_type_helper(full_seq, tasks))
	return tasks

def remove_seq_type(tasks, sample_idx):
	seq_lst = list(sample_idx['train'].keys())
	for task in tasks:
		remove = []
		for seq in task:
			if seq not in seq_lst:
				remove.append(seq)
		for seq in remove:
			task.remove(seq)
	return tasks

def on_task_update(model, task_id, seq_lst, batch_size, gpu_id):
	model.train()
	model.zero_grad()

	with open(os.path.join(mm_dir, args.sample_idx_fn), 'rb') as f:
		sample_idx = pickle.load(f)

	predefined_sample_weights = dict(zip([str(x) for x in range(1,6)], [1.]*5))
	# print(predefined_sample_weights)
	print('creating data loaders')
	dl_lst, _, _ = dl.mm_batchgen(args.full_seq, seq_lst, 
		mm_image_dir, mm_text_dir, mm_video_dir, mm_video_process_dir,
		predefined_sample_weights, sample_idx, 
		num_workers=args.n_workers, batch_size=batch_size,
		seq_count=False, drop_last_tr=False, data_seed=args.data_seed)

	seq_sizes = [len(dataset) for dataset in dl_lst]
	dl_idx = []
	for i in range(len(seq_sizes)):
		dl_idx.extend([i]*seq_sizes[i])

	np.random.shuffle(dl_idx)

	DLS = [iter(cycle(tr_dl)) for tr_dl in dl_lst]

	print('len dl_idx:', len(dl_idx))
	
	for i in dl_idx:
		# print(seq_lst[i])
		DL = DLS[i]
		batch = next(DL)
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
		structured=None
				
		_, loss, _, _, _ = r.mm(model, imgs, text, videos, structured, targets=labels, sample_weights=sample_weights)
		loss.backward()

	fisher_dict[task_id] = {}
	optpar_dict[task_id] = {}

	for name, param in model.named_parameters():
		if param.grad is not None:
			optpar_dict[task_id][name] = param.data.clone()
			fisher_dict[task_id][name] = param.grad.data.clone().pow(2)
		

if __name__ == '__main__':
	# print('setting seeds:', int(args.torch_seed), int(args.random_seed), int(args.numpy_seed))
	# torch.manual_seed(int(args.torch_seed))#(47)
	# random.seed(int(args.random_seed))#(983)
	# np.random.seed(int(args.numpy_seed))#(1024)
	# torch.backends.cudnn.deterministic = True

	# if args.move_out_path and not os.path.exists(args.move_out_path):
	#	 os.makedirs(args.move_out_path)

	# mp.set_start_method('spawn',force=True)
	n_iters = int(args.n_iters)
	n_jobs = int(args.n_jobs)
	tasks=args.tasks
	batch_sizes=args.batch_sizes
	save_interval = int(int(args.save_interval) / n_jobs)
	eval_interval = int(int(args.eval_interval) / n_jobs)

	if not os.path.exists(args.model_save_path):
		os.makedirs(args.model_save_path)

	# if args.move_out_path and not os.path.exists(args.move_out_path):
	#	 os.makedirs(args.move_out_path)

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
		print_log(config, args.model_save_path + '.log')
		print_log(args, args.model_save_path + '.log')
	
	restore_cl = int(args.restore_cl)
	if restore_cl != -1:
		shared_model.restore(args.model_save_path+'/task%s_%s'%(restore_cl-1,len(args.full_seq)-1), 'best/0')
	else:
		restore_cl = 0

	print('send omninet to gpu')
	shared_model=shared_model.to(0)
	print('sent omninet to gpu')
	shared_model.share_memory()
	
	# barrier = mp.Barrier(n_tasks)
	barrier = None
	start = int(restore / n_jobs)

	with open(os.path.join(mm_dir, args.sample_idx_fn), 'rb') as f:
		sample_idx = pickle.load(f)

	all_seq_lst = list(sample_idx['test'].keys())
	cl_tasks = gen_seq_type(args.full_seq)
	cl_tasks = remove_seq_type(cl_tasks, sample_idx)

	# n_iters_lst = [505, 505, 1005, 1005, 1005]
	# eval_interval_lst = [250, 250, 500, 500, 500]
	# save_interval_lst = [250, 250, 500, 500, 500]

	# n_iters_lst = [1505, 1505, 1505, 1505, 1505]
	# eval_interval_lst = [250, 250, 250, 250, 250]
	# save_interval_lst = [250, 250, 250, 250, 250]

	# # n_iters_lst = [1505, 2005, 1005, 1005, 3005]
	# n_iters_lst = [2505, 2505, 2505, 2505, 3505]
	# eval_interval_lst = [100]*5
	# save_interval_lst = [100]*5

	# n_iters_lst = [205, 205, 105, 105, 505]
	# eval_interval_lst = [100]*5
	# save_interval_lst = [100]*5

	# n_iters_lst = [1505, 1505, 1505, 1505, 1505]
	# eval_interval_lst = [300]*5
	# save_interval_lst = [300]*5

	# n_iters_lst = [4005, 4005, 4005, 4005, 5005]
	# eval_interval_lst = [500]*5
	# save_interval_lst = [500]*5

	# n_iters_lst = [5005, 5005, 5005, 5005, 5005]
	n_iters_lst = [10005, 10005, 10005, 10005, 10005]
	eval_interval_lst = [100]*5
	save_interval_lst = [100]*5

	if len(args.ewc_lambda) == 1:
		ewc_lambda = args.ewc_lambda * len(args.full_seq)
	else:
		ewc_lambda = args.ewc_lambda

	if len(args.scale_ewc_lambda) == 1:
		ewc_scales = args.scale_ewc_lambda * n_iters
	else:
		ewc_scales = args.scale_ewc_lambda

	if len(args.n_warmup_steps) == 1:
		n_warmup_steps = args.n_warmup_steps * len(args.full_seq)
	else:
		n_warmup_steps = args.n_warmup_steps

	for k in range(restore_cl, n_iters):
		for task_category_id, seq_lst in enumerate(cl_tasks):
			task_id = k*len(args.full_seq) + task_category_id
			counters = [Counter(restore) for i in range(len(tasks))]

			# train
			print('training task:', task_category_id)
			notest = True
			test = False
			test_on_val = False
			gpu_id=0
			model_save_path = args.model_save_path + '/task%s_%s'%(k,task_category_id)
			train(shared_model, tasks[0], batch_sizes[0],
					 int(n_iters_lst[task_category_id] / n_jobs),
					 gpu_id, start, restore, counters[0], barrier,
					 args.n_workers,
					 save_interval_lst[task_category_id],
					 eval_interval_lst[task_category_id],
					 args.eval_first,
					 True,
					 args.peripherals_type, args.conf_type,
					 model_save_path, args.move_out_path, args.save_best, args.testing,
					 args.sample_idx_fn, seq_lst, args.rewarding, 
					 args.random_seq, notest, test, test_on_val,
					 args.use_sgd,
					 task_id, ewc_lambda[task_category_id]*ewc_scales[k], 
					 args.full_seq, None, None) #int(args.all_seed)+k) 
					 # use args.all_seed+k to set a different seed for dataloader in a different pass
					 # set to None for same seed

			# use the best model
			shared_model.restore(model_save_path, 'best/0')
			
			print('on_task_update task:', task_id)
			start_time = time.time()
			on_task_update(shared_model, task_category_id, seq_lst, batch_sizes[0], gpu_id=0)
			print('on_task_update: %.2f'%(time.time() - start_time))

			# test 
			for test, test_on_val in [(True, True), (True, False)]:
				# test = True
				notest = False
				counting_reward_dict = {}

				if k == 0:
					test_tasks = cl_tasks[:(task_category_id+1)]
				else:
					test_tasks = cl_tasks

				for task_category_id_test, seq_lst_test in enumerate(test_tasks):
					print('testing task:', task_category_id_test)
					counters = [Counter(restore) for i in range(len(tasks))]
					train(shared_model, tasks[0], batch_sizes[0],
							 int(n_iters_lst[task_category_id_test] / n_jobs),
							 gpu_id, start, restore, counters[0], barrier,
							 args.n_workers,
							 save_interval_lst[task_category_id_test],
							 eval_interval_lst[task_category_id_test],
							 args.eval_first,
							 True,
							 args.peripherals_type, args.conf_type,
							 model_save_path, args.move_out_path, args.save_best, args.testing,
							 args.sample_idx_fn, seq_lst_test, args.rewarding,
							 args.random_seq, notest, test, test_on_val,
							 args.use_sgd,
							 task_category_id_test, ewc_lambda[task_category_id]*ewc_scales[k], 
							 args.full_seq,
							 counting_reward_dict, None)

			with open(args.model_save_path + '/current_iterations.pkl', 'wb') as f:
				pickle.dump(current_iterations, f)

			with open(args.model_save_path + '/fisher_dict.pkl', 'wb') as f:
				pickle.dump(fisher_dict, f)
			with open(args.model_save_path + '/optpar_dict.pkl', 'wb') as f:
				pickle.dump(optpar_dict, f)

	# shared_model.save(args.model_save_path, 'last')
