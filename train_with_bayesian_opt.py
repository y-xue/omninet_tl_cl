from skopt import gp_minimize
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern

from bayesian_opt import *

import os
import subprocess
import json
import re
import time
import random
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='train with bayesian optimization.')
parser.add_argument('--gpu_id', default='0', type=str, help='which gpu to use.')
parser.add_argument('--out_path', default='mm_nodup_data_random_seq_detect_overfitting_with_tr_acc_stop_overfitted_seq_0.0001ot_1000os_dropout_0.5_0.25', help='output path')
parser.add_argument('--seed', default=1029, type=int, help='seed')
parser.add_argument('--bo_n_iter', default=10, type=int, help='number of BO iterations')
parser.add_argument('--n_iter', default=4000, type=int, help='number of iterations of omninet training')
parser.add_argument('--test_bo_iter', default=None, type=int, help='test a given bo iter')
parser.add_argument('--xi', default=0.01, type=float, help='exploitation-exploration parameter')
parser.add_argument('--restore', default=-1, type=int, help='Step from which to restore model training')
parser.add_argument('--sample_weights', default=None, type=str, help='sample weights separated by comma')
parser.add_argument('--data_seed', default=615, type=int, help='data seed')
parser.add_argument('--init_lr', default=0.02, type=float, help='init_lr')
parser.add_argument('--n_warmup_steps', default=16000, type=int, help='n_warmup_steps for ScheduledOptim')

args = parser.parse_args()

def set_seed(seed=615):
    random.seed(seed)
    np.random.seed(seed)
    
set_seed(args.seed)

class Trainer(object):
	def __init__(self):
		# cuda_device=0
		# server_name=wcdl1
		self.xi = args.xi
		self.save_intvl=500
		self.eval_intvl=500
		self.bs=16
		self.n_iter=args.n_iter
		self.seq='ITTIV'
		self.task='mm_ITV'
		self.n_gpus=1
		self.init_lr=args.init_lr
		self.all_seed=args.seed #(1029 47 816 21 219 222 628 845 17 531 635)
		self.data_seed=args.data_seed
		self.n_warmup_steps=args.n_warmup_steps
		self.overfitting_threshold = 0.0001
		self.start_detect_overfitting_iter = 1000

		self.out_path = args.out_path

		self.move_out='/files/yxue/research/allstate/out'
		self.data_path='/files/yxue/research/allstate/data/synthetic_mm_cifar_imdb_hmdb'
		self.bo_iter_output_path='mm_bo/'+ self.out_path + '/%s_%sbs_%sit_allseed%s'
		self.log_fn = os.path.join(self.move_out, 
			'mm_bo/'+ self.out_path + '/%s_%sbs_%sit_allseed%s.log')%(
			self.seq, self.bs, self.n_iter, self.all_seed)

		self.bo_iter = 0
		self.best_bo_iter = 0

		self.my_env = os.environ.copy()
		self.my_env["CUDA_VISIBLE_DEVICES"] = '%s'%(args.gpu_id)

		# print(self.log_fn)

		if not os.path.exists(os.path.join(self.data_path, 'bo/%s'%self.out_path)):
			os.makedirs(os.path.join(self.data_path, 'bo/%s'%self.out_path))

		if not os.path.exists(os.path.join(self.move_out, 'mm_bo/'+ self.out_path)):
			os.makedirs(os.path.join(self.move_out, 'mm_bo/'+ self.out_path))
		# subprocess.run(['python',
		# 		'train.py',
		# 		'%s'%n_iter,
		# 		'%s'%task,
		# 		'%s'%bs,
		# 		'--n_gpus',
		# 		'%s'%n_gpus,
		# 		'--save_interval',
		# 		'%s'%save_intvl,
		# 		'--eval_interval',
		# 		'%s'%eval_intvl,
		# 		'--model_save_path',
		# 		'/out/mm/%s_dataseed1024_sample_weights%s_%sbs_%sit_allseed%s'%(seq, sw, bs, n_iter, all_seed),
		# 		'--move_out_path',
		# 		'%s/mm/tmp_%s_dataseed1024_sample_weights%s_%sbs_%sit_allseed%s'%(move_out, seq, sw, bs, n_iter, all_seed),
		# 		'--sample_weights_fn',
		# 		'sample_weights_%s_%s.json'%(seq, sw),
		# 		'--peripherals_type',
		# 		'timeline',
		# 		'--conf_type',
		# 		'timeline',
		# 		'--all_seed',
		# 		'%s'%all_seed], stdout=subprocess.PIPE, env=my_env)
		# 		# '<',
		# 		# '/dev/null',
		# 		# '>',
		# 		# '/dev/null',
		# 		# '2>&1;'])

		# CUDA_VISIBLE_DEVICES=${cuda_device} python train.py ${n_iter} ${task} ${bs} --n_gpus ${n_gpus} \
		# 	--save_interval ${save_intvl} --eval_interval ${eval_intvl} \
		# 	--model_save_path /out/mm/${seq}_dataseed1024_sample_weights${sw}_${bs}bs_${n_iter}it_$((all_seed[$i])) \
		# 	--move_out_path ${move_out}/mm/${seq}_dataseed1024_sample_weights${sw}_${bs}bs_${n_iter}it_$((all_seed[$i])) \
		# 	--sample_weights_fn sample_weights_${seq}_${sw}.json --peripherals_type timeline --conf_type timeline \
		# 	--all_seed $((all_seed[$i])) > /mm_${seq}_dataseed1024_sample_weights${sw}_${bs}bs_${n_iter}it_$((all_seed[$i])).log;

	def read_best_val_scores(self, move_out_path, score='reward'):
		with open('%s.log'%move_out_path, 'r') as f:
			log = f.read()
		
		val_res = re.findall(r'Step (\d+), mm total validation Accuracy (\d+\.\d+) \%, reward: (\d+\.\d+)', log)
		val_acc = np.array([float(x[1]) for x in val_res])
		val_reward = np.array([float(x[2]) for x in val_res])
		val_step = np.array([float(x[0]) for x in val_res])

		if score == 'reward':
			return max(val_reward)
		return max(val_acc)

	def read_test_scores(self, move_out_path, score='reward'):
		with open('%s.log'%move_out_path, 'r') as f:
			log = f.read()

		test_acc_res = re.findall(r'.*mm total test Accuracy (\d+\.\d+) \%, reward: (\d+\.\d+)', log)[0]
		test_acc = float(test_acc_res[0])
		test_reward = float(test_acc_res[1])

		if score == 'reward':
			return test_reward
		return test_acc

	def write_sample_weights(self, sample_weights, fn):
		# print(sample_weights)
		with open(os.path.join(self.data_path,fn), 'w') as f: 
			# print(dict(zip(range(1,len(self.seq)+1), sample_weights)))
			json.dump(dict(zip(range(1,len(self.seq)+1), sample_weights)), f)

	def total_val_reward_init(self, sample_weights, i):
		self.bo_iter = i
		return self.total_val_reward(sample_weights)

	def total_val_reward(self, sample_weights, test=False):
		# print(sample_weights)

		model_save_path = os.path.join('out', self.bo_iter_output_path+'_boiter%s')%(self.seq, self.bs, self.n_iter, self.all_seed, self.bo_iter)
		move_out_path = os.path.join(self.move_out, self.bo_iter_output_path+'_boiter%s')%(self.seq, self.bs, self.n_iter, self.all_seed, self.bo_iter)

		if os.path.exists('%s.log'%move_out_path):
			# print('reading res from ', '%s.log'%move_out_path)
			# print('log exists')
			return self.read_best_val_scores(move_out_path)

		# print(sample_weights)

		fn = 'bo/%s/sample_weights_boiter_%s'%(self.out_path, self.bo_iter)
		
		# for general cases
		# self.write_sample_weights(sample_weights, fn)

		# # for ITTIV nodup data
		# self.write_sample_weights(np.hstack((sample_weights,[1])), fn)

		# for ITTIV all nodup data
		self.write_sample_weights(sample_weights, fn)

		training_cmd = ['python',
			'train.py',
			'%s'%self.n_iter,
			'%s'%self.task,
			'%s'%self.bs,
			'--n_gpus',
			'%s'%self.n_gpus,
			'--save_interval',
			'%s'%self.save_intvl,
			'--restore',
			'%s'%args.restore,
			'--eval_interval',
			'%s'%self.eval_intvl,
			'--save_best',
			'--model_save_path',
			model_save_path,
			'--move_out_path',
			move_out_path,
			'--init_lr',
			'%s'%self.init_lr,
			'--n_warmup_steps',
			'%s'%self.n_warmup_steps,
			'--sample_idx_fn', 
			'sample_idx_ITTIV_pT.7_no_dup_all_img_text_video_seed10', 
			'--sample_weights_fn',
			fn,
			'--peripherals_type',
			'timeline',
			'--conf_type',
			'timeline',
			'--all_seed',
			'%s'%self.all_seed,
			'--data_seed',
			'%s'%self.data_seed,
			'--rewarding',
			'--pick_seq', 
			'random',
			'--notest']

		# detect overfitting
		# training_cmd = ['python',
		# 	'train.py',
		# 	'%s'%self.n_iter,
		# 	'%s'%self.task,
		# 	'%s'%self.bs,
		# 	'--n_gpus',
		# 	'%s'%self.n_gpus,
		# 	'--save_interval',
		# 	'%s'%self.save_intvl,
		# 	'--eval_interval',
		# 	'%s'%self.eval_intvl,
		# 	'--save_best',
		# 	'--model_save_path',
		# 	model_save_path,
		# 	'--move_out_path',
		# 	move_out_path,
		# 	'--sample_idx_fn', 
		# 	'sample_idx_ITTIV_pT.7_no_dup_all_img_text_video_seed10', 
		# 	'--sample_weights_fn',
		# 	fn,
		# 	'--peripherals_type',
		# 	'timeline',
		# 	'--conf_type',
		# 	'timeline',
		# 	'--all_seed',
		# 	'%s'%self.all_seed,
		# 	'--rewarding',
		# 	'--pick_seq', 
		# 	'random',
		# 	'--overfitting_threshold',
		# 	'%s'%self.overfitting_threshold,
		# 	'--overfitting_start',
		# 	'%s'%self.start_detect_overfitting_iter,
		# 	'--detect_overfitting_with_tr_acc',
		# 	'--notest']

		# training_cmd = ['python',
		# 	'train.py',
		# 	'%s'%self.n_iter,
		# 	'%s'%self.task,
		# 	'%s'%self.bs,
		# 	'--n_gpus',
		# 	'%s'%self.n_gpus,
		# 	'--save_interval',
		# 	'%s'%self.save_intvl,
		# 	'--eval_interval',
		# 	'%s'%self.eval_intvl,
		# 	'--model_save_path',
		# 	model_save_path,
		# 	'--move_out_path',
		# 	move_out_path,
		# 	'--save_best',
		# 	'--norm_weights', 
		# 	'--scale_weights', 
		# 	'--sample_idx_fn', 
		# 	'sample_idx_ITTIV_pT.7_no_dup_img_text_video_seed10', 
		# 	'--sample_weights_fn',
		# 	fn,
		# 	'--peripherals_type',
		# 	'timeline',
		# 	'--conf_type',
		# 	'timeline',
		# 	'--all_seed',
		# 	'%s'%self.all_seed,
		# 	'--rewarding', 
		# 	'--pick_sample_by_intsance_id',
		# 	'--pick_seq', 
		# 	'sequential',
		# 	'--notest']

		output = subprocess.run(training_cmd, env=self.my_env, encoding='utf-8', 
			stdout=subprocess.PIPE)

		# # print(output.stdout.read())
		# for line in output.stdout.split('\n'):
		# 	print(line)

		time.sleep(10)

		return self.read_best_val_scores(move_out_path)

	def test(self):
		print('test model at bo_iter %s'%self.best_bo_iter)
		move_out_path = os.path.join(self.move_out, self.bo_iter_output_path+'_boiter%s')%(self.seq, self.bs, self.n_iter, self.all_seed, self.best_bo_iter)

		fn = 'bo/%s/sample_weights_boiter_%s'%(self.out_path, self.best_bo_iter)

		training_cmd = ['python',
			'train.py',
			'%s'%self.n_iter,
			'%s'%self.task,
			'%s'%self.bs,
			'--n_gpus',
			'%s'%self.n_gpus,
			'--save_interval',
			'%s'%self.save_intvl,
			'--eval_interval',
			'%s'%self.eval_intvl,
			'--save_best',
			'--model_save_path',
			move_out_path,
			'--move_out_path',
			move_out_path,
			'--init_lr',
			'%s'%self.init_lr,
			'--n_warmup_steps',
			'%s'%self.n_warmup_steps,
			'--sample_idx_fn', 
			'sample_idx_ITTIV_pT.7_no_dup_all_img_text_video_seed10', 
			'--sample_weights_fn',
			fn,
			'--peripherals_type',
			'timeline',
			'--conf_type',
			'timeline',
			'--all_seed',
			'%s'%self.all_seed,
			'--data_seed',
			'%s'%self.data_seed,
			'--rewarding',
			'--pick_seq', 
			'random',
			'--test']

		# # detect overfitting
		# training_cmd = ['python',
		# 	'train.py',
		# 	'%s'%self.n_iter,
		# 	'%s'%self.task,
		# 	'%s'%self.bs,
		# 	'--n_gpus',
		# 	'%s'%self.n_gpus,
		# 	'--save_interval',
		# 	'%s'%self.save_intvl,
		# 	'--eval_interval',
		# 	'%s'%self.eval_intvl,
		# 	'--save_best',
		# 	'--model_save_path',
		# 	move_out_path,
		# 	'--move_out_path',
		# 	move_out_path, 
		# 	'--sample_idx_fn', 
		# 	'sample_idx_ITTIV_pT.7_no_dup_img_text_video_seed10', 
		# 	'--sample_weights_fn',
		# 	fn,
		# 	'--peripherals_type',
		# 	'timeline',
		# 	'--conf_type',
		# 	'timeline',
		# 	'--all_seed',
		# 	'%s'%self.all_seed,
		# 	'--rewarding', 
		# 	'--pick_seq', 
		# 	'random',
		# 	'--overfitting_threshold',
		# 	'%s'%self.overfitting_threshold,
		# 	'--overfitting_start',
		# 	'%s'%self.start_detect_overfitting_iter,
		# 	'--detect_overfitting_with_tr_acc',
		# 	'--test']

		output = subprocess.run(training_cmd, env=self.my_env, encoding='utf-8', 
			stdout=subprocess.PIPE)

		# # print(output.stdout.read())
		# for line in output.stdout.split('\n'):
		# 	print(line)
		test_reward = self.read_test_scores(move_out_path)
		
		with open(self.log_fn, 'a') as f:
			print('test reward at bo_iter %s:'%self.best_bo_iter, test_reward, file=f)


	def train(self):

		# # # default init weights for ITTIV
		# # sample_weights0=[1,1,1,1,1]
		# # sample_weights1=[0.9, 0.05, 0.02, 0.018, 0.012]

		# # weights for ITTIV nodup data
		# # where there is no sample having 5 modes.
		# # sample_weights1 are from experiments with 
		# # --norm_weights and --scale_weights.
		# # Values are scaled so that they sum up to 4
		# sample_weights0=[1,1,1,1]
		# # sample_weights1=[1.12044429, 1.11438296, 1.11279222, 0.65238053]
		# # sample_weights1=[4.,3.,2.,1.]
		# random_init_sample_weights=[list(np.random.uniform(0.01,4,4)) for _ in range(3)]

		# best weights from experiments with 
		# --norm_weights and --scale_weights.
		# Values are scaled so that they sum up to 5
		# sample_weights1=[1.16595927, 1.07395726, 1.06814742, 1.06662267, 0.62531338]

		t = len(self.seq)
		# sample_weights0=[0.25]*t
		# sample_weights0=[1.0]*t
		
		sample_weights0 = [float(x) for x in args.sample_weights.split(',')]
		# random_init_sample_weights=[list(np.random.uniform(1e-8,1,t)) for _ in range(3)]


		X_init = [sample_weights0]
		# X_init.extend(random_init_sample_weights)
		X_init = np.array(X_init)
		Y_init = np.array([self.total_val_reward_init(x, i-X_init.shape[0]) for i,x in enumerate(X_init)]).reshape(-1,1)
		# Y_init = np.array([3.476274, 3.416073]).reshape(-1,1)
		# sample_weights2 = [0.35567414, 0.01606683, 0.41599741, 1.94683089]
		# sample_weights3 = [1.938421240043022, 1.8494988671951496, 0.012674311372760452, 1.4889111764365908]
		# X_init = np.array([sample_weights0, sample_weights1, sample_weights2, sample_weights3])
		# Y_init = np.array([3.353207, 3.254208, 3.352601, 3.365584])

		noise = 0.2
		bounds = np.array([[1e-8, 1.0]]*t)

		# Gaussian process with Mat??rn kernel as surrogate model
		m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
		gpr = GaussianProcessRegressor(kernel=m52, alpha=noise**2)

		# Initialize samples
		X_sample = X_init
		Y_sample = Y_init

		# Number of iterations
		n_iter = args.bo_n_iter

		# plt.figure(figsize=(12, n_iter * 3))
		# plt.subplots_adjust(hspace=0.4)

		with open(self.log_fn, 'a') as f:
			print('X_init:', X_init, file=f)
			print('Y_init:', Y_init, file=f)

		best_Y = 0
		# best_Y = max(Y_init)
		for i in range(len(Y_init)):
			if Y_init[i] > best_Y:
				best_Y = Y_init[i]
				self.best_bo_iter = i-len(Y_init)

		with open(self.log_fn, 'a') as f:
			print('best initial Y:', best_Y, file=f)
			print('best initial iter:', self.best_bo_iter, file=f)

		for i in range(n_iter):
			self.bo_iter = i
			# Update Gaussian process with existing samples
			gpr.fit(X_sample, Y_sample)

			# Obtain next sampling point from the acquisition function (expected_improvement)
			X_next = propose_location(expected_improvement, X_sample, Y_sample, gpr, bounds, xi=self.xi)

			# print('next X:', X_next)
			#     print(X_next.shape)

			# Obtain next noisy sample from the objective function
			Y_next = self.total_val_reward(X_next)

			# print('next Y:', Y_next)
			# print(Y_next.shape)

			with open(self.log_fn, 'a') as f:
				print('bo_iter ', i, file=f)
				print('next X:', X_next, file=f)
				print('next Y:', Y_next, file=f)

			print('bo_iter ', i)
			print('next X:', X_next)
			print('next Y:', Y_next)

			if Y_next > best_Y:
				best_Y = Y_next
				self.best_bo_iter = i

			# # Plot samples, surrogate function, noise-free objective and next sampling location
			# plt.subplot(n_iter, 2, 2 * i + 1)
			# plot_approximation(gpr, X, Y, X_sample, Y_sample, X_next, show_legend=i==0)
			# plt.title(f'Iteration {i+1}')

			# plt.subplot(n_iter, 2, 2 * i + 2)
			# plot_acquisition(X, expected_improvement(X, X_sample, Y_sample, gpr), X_next, show_legend=i==0)

			# Add sample to previous samples
			X_sample = np.vstack((X_sample, X_next))
			Y_sample = np.vstack((Y_sample, Y_next))

trainer = Trainer()

if args.test_bo_iter is not None:
	trainer.best_bo_iter = args.test_bo_iter
	trainer.test()
else:
	trainer.train()
	trainer.test()
	# trainer.best_bo_iter = -4
	# trainer.test()

# trainer.best_bo_iter = -1
# trainer.test()

# m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
# gpr = GaussianProcessRegressor(kernel=m52, alpha=noise**2)

# r = gp_minimize(lambda x: -total_val_reward(x), 
#                 bounds,
#                 base_estimator=gpr,
#                 acq_func='EI',      # expected improvement
#                 xi=0.01,            # exploitation-exploration trade-off
#                 n_calls=10,         # number of iterations
#                 n_random_starts=0,  # initial samples are provided
#                 x0=X_init, # initial samples
#                 y0=(-Y_init).tolist())

# # # Fit GP model to samples for plotting results
# gpr.fit(r.x_iters, -r.func_vals)


"""
# Dense grid of points within bounds
X = np.arange(bounds[:, 0], bounds[:, 1], 0.01).reshape(-1, 1)
# Noise-free objective function values at X 
Y = total_val_reward(X)
# Plot the fitted model and the noisy samples
plot_approximation(gpr, X, Y, r.x_iters, -r.func_vals, show_legend=True)

"""