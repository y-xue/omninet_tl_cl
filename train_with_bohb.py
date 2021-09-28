import logging
logging.basicConfig(level=logging.WARNING)

import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
# from hpbandster.examples.commons import MyWorker
from BOHB_worker import MyWorker

import numpy as np
import random
import os

parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
parser.add_argument('--min_budget',   type=float, help='Minimum budget used during the optimization.',    default=9)
parser.add_argument('--max_budget',   type=float, help='Maximum budget used during the optimization.',    default=243)
parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=4)
parser.add_argument('--seed', type=int, help='seed', default=1029)
parser.add_argument('--lr', type=int, help='lr', default=0.01)
parser.add_argument('--n_warmup_steps', type=int, help='n_warmup_steps', default=5000)
parser.add_argument('--out_path', default='mm_nodup_data_random_seq_detect_overfitting_with_tr_acc_stop_overfitted_seq_0.0001ot_1000os_dropout_0.5_0.25', help='output path')
parser.add_argument('--gpu_id', default='0', type=str, help='which gpu to use.')
parser.add_argument('--port', default=9090, type=int, help='port')

args=parser.parse_args()

def set_seed(seed=615):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(args.seed)

# Step 1: Start a nameserver
# Every run needs a nameserver. It could be a 'static' server with a
# permanent address, but here it will be started for the local machine with the default port.
# The nameserver manages the concurrent running workers across all possible threads or clusternodes.
# Note the run_id argument. This uniquely identifies a run of any HpBandSter optimizer.
print('starting nameserver')
NS = hpns.NameServer(run_id=str(args.seed), host='127.0.0.1', port=args.port)
NS.start()
print('NameServer started')

# Step 2: Start a worker
# Now we can instantiate a worker, providing the mandatory information
# Besides the sleep_interval, we need to define the nameserver information and
# the same run_id as above. After that, we can start the worker in the background,
# where it will wait for incoming configurations to evaluate.
w = MyWorker(nameserver='127.0.0.1',nameserver_port=args.port,run_id=str(args.seed),seed=args.seed,out_path=args.out_path,gpu_id=args.gpu_id,lr=args.lr,n_warmup_steps=args.n_warmup_steps)
w.run(background=True)

print('worker running')

# Step 3: Run an optimizer
# Now we can create an optimizer object and start the run.
# Here, we run BOHB, but that is not essential.
# The run method will return the `Result` that contains all runs performed.
bohb = BOHB(  configspace = w.get_configspace(args.seed),
              run_id = str(args.seed), nameserver='127.0.0.1',nameserver_port=args.port,
              min_budget=args.min_budget, max_budget=args.max_budget
           )
print('run bohb')
res = bohb.run(n_iterations=args.n_iterations)

# w.best_bo_iter = '-'.join([str(x) for x in res.get_incumbent_id()])
# w.test()

# Step 4: Shutdown
# After the optimizer run, we must shutdown the master and the nameserver.
bohb.shutdown(shutdown_workers=True)
NS.shutdown()

# Step 5: Analysis
# Each optimizer returns a hpbandster.core.result.Result object.
# It holds informations about the optimization run like the incumbent (=best) configuration.
# For further details about the Result object, see its documentation.
# Here we simply print out the best config and some statistics about the performed runs.
id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()
print('incumbent:', incumbent)

print('Best found configuration:', id2config[incumbent]['config'])
print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
print('A total of %i runs where executed.' % len(res.get_all_runs()))
print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in res.get_all_runs()])/args.max_budget))


