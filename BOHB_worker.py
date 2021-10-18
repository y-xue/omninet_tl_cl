import numpy as np
import time

import ConfigSpace as CS
from hpbandster.core.worker import Worker

import os
import subprocess
import json
import re
import random

class MyWorker(Worker):

    def __init__(self, *args, seed=1029, out_path='mm_random_seq_dropout_0.5_0.25_allseed219', gpu_id=0, xi=0.01, lr=0.01, n_warmup_steps=5000, large_seq_rewarding=False, restore_last=True, **kwargs):
        super().__init__(*args, **kwargs)

        self.xi = xi
        self.save_intvl=500
        self.eval_intvl=500
        self.lr=lr
        self.bs=16
        # self.n_iter=4005
        self.seq='ITTIV'
        self.task='mm_ITV'
        self.n_gpus=1
        self.all_seed=seed #(1029 47 816 21 219 222 628 845 17 531 635)
        self.n_warmup_steps=n_warmup_steps
        self.restore_last = restore_last
        self.large_seq_rewarding = large_seq_rewarding
        self.overfitting_threshold = 0.0001
        self.start_detect_overfitting_iter = 1000

        self.out_path = out_path

        # self.move_out='/Users/ye/Documents/research/allstate/out'
        # self.data_path='/Users/ye/Documents/research/allstate/data/synthetic_mm_cifar_imdb_hmdb'
        self.move_out='/files/yxue/research/allstate/out'
        self.data_path='/files/yxue/research/allstate/data/synthetic_mm_cifar_imdb_hmdb'
        self.bo_iter_output_path='mm_bo/'+ self.out_path + '/%s_%sbs'
        
        self.bo_iter = 0
        self.best_bo_iter = 0

        self.my_env = os.environ.copy()
        self.my_env["CUDA_VISIBLE_DEVICES"] = '%s'%(gpu_id)

        # print(self.log_fn)

        if not os.path.exists(os.path.join(self.data_path, 'bo/%s'%self.out_path)):
            os.makedirs(os.path.join(self.data_path, 'bo/%s'%self.out_path))

        if not os.path.exists(os.path.join(self.move_out, 'mm_bo/'+ self.out_path)):
            os.makedirs(os.path.join(self.move_out, 'mm_bo/'+ self.out_path))

    def read_best_val_scores(self, move_out_path, score='reward'):
        with open('%s.log'%move_out_path, 'r') as f:
            log = f.read()
        
        val_res = re.findall(r'Step (\d+),.*total validation Accuracy (\d+\.\d+) \%, reward: (\d+\.\d+)', log)
        if len(val_res) == 0:
            return None
        val_acc = np.array([float(x[1]) for x in val_res])
        val_reward = np.array([float(x[2]) for x in val_res])
        val_step = np.array([float(x[0]) for x in val_res])

        if score == 'reward':
            return max(val_reward[val_step<self.n_iter])
        return max(val_acc[val_step<self.n_iter])

    def read_test_scores(self, move_out_path, score='reward'):
        with open('%s.log'%move_out_path, 'r') as f:
            log = f.read()

        test_acc_res = re.findall(r'Step (\d+),.*total test Accuracy (\d+\.\d+) \%, reward: (\d+\.\d+)', log)#[0]
        if len(test_acc_res) == 0:
            return None
        steps = [float(test_acc_res[i][0]) for i in range(len(test_acc_res))]
        test_acc_rewards = [(float(test_acc_res[i][1]), float(test_acc_res[i][2])) for i in range(len(test_acc_res))]
        res = dict(zip(steps, test_acc_rewards))
        # test_acc = float(test_acc_res[0])
        # test_reward = float(test_acc_res[1])

        test_reward = test_acc = None
        for step in res:
            if step + 1 == self.n_iter:
                test_acc, test_reward = res[step]
                break

        if score == 'reward':
            return test_reward
        return test_acc

    def write_sample_weights(self, sample_weights, fn):
        # print(sample_weights)
        with open(os.path.join(self.data_path,fn), 'w') as f: 
            # print(dict(zip(range(1,len(self.seq)+1), sample_weights)))
            json.dump(dict(zip(range(1,len(self.seq)+1), sample_weights)), f)

    def total_val_reward(self, sample_weights):
        # print(sample_weights)

        # model_save_path = os.path.join('out', self.bo_iter_output_path+'_boiter%s')%(self.seq, self.bs, self.n_iter, self.all_seed, self.bo_iter)
        move_out_path = os.path.join(self.move_out, self.bo_iter_output_path+'_boiter%s')%(self.seq, self.bs, self.bo_iter)

        if os.path.exists('%s.log'%move_out_path):
            # print('reading res from ', '%s.log'%move_out_path)
            # print('log exists')
            best_val_reward, test_reward = self.read_best_val_scores(move_out_path), self.read_test_scores(move_out_path)
            if test_reward is not None:
                return best_val_reward, test_reward

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
            '--eval_interval',
            '%s'%self.eval_intvl,
            '--init_lr',
            '%s'%self.lr,
            '--n_warmup_steps',
            '%s'%self.n_warmup_steps,
            '--save_best',
            '--model_save_path',
            move_out_path,
            '--move_out_path',
            move_out_path,
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
            '--rewarding',
            '--pick_seq', 
            'random']

        if self.restore_last:
            training_cmd.append('--restore_last')

        if self.large_seq_rewarding:
            training_cmd.append('--large_seq_rewarding')

        output = subprocess.run(training_cmd, env=self.my_env, encoding='utf-8', 
            stdout=subprocess.PIPE)

        time.sleep(5)

        best_val_reward = self.read_best_val_scores(move_out_path)
        test_reward = self.read_test_scores(move_out_path)

        return best_val_reward, test_reward

    # def test(self):
    #     with open(self.log_fn, 'a') as f:
    #         print('test model at bo_iter %s'%self.best_bo_iter, file=f)
    #     move_out_path = os.path.join(self.move_out, self.bo_iter_output_path+'_boiter%s')%(self.seq, self.bs, self.n_iter, self.all_seed, self.best_bo_iter)

    #     fn = 'bo/%s/sample_weights_boiter_%s'%(self.out_path, self.best_bo_iter)

    #     training_cmd = ['python',
    #         'train.py',
    #         '%s'%self.n_iter,
    #         '%s'%self.task,
    #         '%s'%self.bs,
    #         '--n_gpus',
    #         '%s'%self.n_gpus,
    #         '--save_interval',
    #         '%s'%self.save_intvl,
    #         '--eval_interval',
    #         '%s'%self.eval_intvl,
    #         '--save_best',
    #         '--model_save_path',
    #         move_out_path,
    #         '--move_out_path',
    #         move_out_path,
    #         '--sample_idx_fn', 
    #         'sample_idx_ITTIV_pT.7_no_dup_all_img_text_video_seed10', 
    #         '--sample_weights_fn',
    #         fn,
    #         '--peripherals_type',
    #         'timeline',
    #         '--conf_type',
    #         'timeline',
    #         '--all_seed',
    #         '%s'%self.all_seed,
    #         '--rewarding',
    #         '--pick_seq', 
    #         'random',
    #         '--test']

    #     output = subprocess.run(training_cmd, env=self.my_env, encoding='utf-8', 
    #         stdout=subprocess.PIPE)

    #     # # print(output.stdout.read())
    #     # for line in output.stdout.split('\n'):
    #     #   print(line)
    #     test_reward = self.read_test_scores(move_out_path)
        
    #     with open(self.log_fn, 'a') as f:
    #         print('test reward at bo_iter %s:'%self.best_bo_iter, test_reward, file=f)

    #     time.sleep(5)

    def compute(self, config_id, config, budget, **kwargs):
        """
        Simple example for a compute function
        The loss is just a the config + some noise (that decreases with the budget)

        For dramatization, the function can sleep for a given interval to emphasizes
        the speed ups achievable with parallel workers.

        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train

        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """
        self.n_iter=int(budget//self.eval_intvl*self.eval_intvl+5)

        self.log_fn = os.path.join(self.move_out, 
            'mm_bo/'+ self.out_path + '/%s_%sbs_allseed%s.log')%(
            self.seq, self.bs, self.all_seed)

        with open(self.log_fn, 'a') as f:
            print('config_id:', config_id, file=f)
            print('budget:', self.n_iter, file=f)
            print('config:', config, file=f)

        self.bo_iter = '-'.join([str(x) for x in config_id])
        X_next = np.array([config['w1'],config['w2'],config['w3'],config['w4'],config['w5']])
        val_reward, test_reward = self.total_val_reward(X_next)
        res = -1*val_reward
        # if self.n_iter > 4000:

        #     self.log_fn = os.path.join(self.move_out, 
        #         'mm_bo/'+ self.out_path + '/%s_%sbs_allseed%s.log')%(
        #         self.seq, self.bs, self.all_seed)

        #     with open(self.log_fn, 'a') as f:
        #         print('test config_id:', config_id, file=f)
        #         print('test budget:', self.n_iter, file=f)
        #         print('test config:', config, file=f)

        #     self.bo_iter = '-'.join([str(x) for x in config_id])
        #     # X_next = np.array([config['w1'],config['w2'],config['w3'],config['w4'],config['w5']])
        #     # res = -1*self.total_val_reward(X_next)
        #     self.best_bo_iter = self.bo_iter
        #     self.test()
        # res=0

        with open(self.log_fn, 'a') as f:
            print('best val reward:', val_reward, file=f)
            print('test reward:', test_reward, file=f)
            print('\n', file=f)

        return({
                    'loss': float(res),  # this is the a mandatory field to run hyperband
                    'info': res  # can be used for any user-defined information - also mandatory
                })

    @staticmethod
    def get_configspace(seed):
        # random.seed(seed)
        # np.random.seed(seed)
        # os.environ['PYTHONHASHSEED'] = str(seed)
    
        config_space = CS.ConfigurationSpace(seed=seed)
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('w1', lower=0, upper=1))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('w2', lower=0, upper=1))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('w3', lower=0, upper=1))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('w4', lower=0, upper=1))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('w5', lower=0, upper=1))
        return(config_space)