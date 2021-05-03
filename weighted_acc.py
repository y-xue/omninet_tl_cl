import numpy as np
import re
import json
import argparse

seq_lst = ['0-1-0', '0-2-0', '1-2-0', '1-3-0', '2-3-0', '2-4-0', 
			'2-5-0', '2-6-0', '0-3-0', '0-4-0', '0-5-0', '0-6-0', 
			'1-0-0', '1-1-0', '1-4-0', '1-5-0', '1-6-0', '1-4-1', 
			'1-6-1', '0-6-1', '2-2-0', '0-5-1', '2-1-0', '2-6-1', 
			'3-3-0', '3-4-0', '3-5-0', '3-6-0', '0-4-1', '2-5-1', 
			'3-2-0', '1-5-1', '2-0-0']

n_total = [3865,3680,2119,2287,599,560,502,306,2826,2324,2087,1254,1133,1316,1989,1767,1046,16,59,85,347,54,48,12,47,46,42,23,13,13,14,40,1]

def get_test_acc(fn):
		with open(fn, 'r') as f:
			log = f.read()
		return [float(x)/100 for x in re.findall(r'.*test.*, Accuracy (\d+\.\d+) \%', log)]

def weighted_acc(log_fn, sample_weights_fn):
	n_correct = (np.array(n_total) * np.array(get_test_acc(log_fn))).round(0)

	with open(sample_weights_fn, 'r') as f:
		sw = json.load(f)

	sw = np.array(list(sw.values()))

	n_total_by_type = [0 for _ in range(10)]
	n_correct_by_type = [0 for _ in range(10)]
	for i in range(len(seq_lst)):
		seq = seq_lst[i]
		n = sum(list(map(int,seq.split('-'))))
		n_total_by_type[n-1] += n_total[i]
		n_correct_by_type[n-1] += n_correct[i]

	return sum(n_correct_by_type*sw)/sum(n_total_by_type*sw)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='OmniNet Evaluation script.')
	parser.add_argument('--log_fn', help='log file name.')
	parser.add_argument('--sample_weights_fn', help='sample weights file name.')
	args = parser.parse_args()
	# fn = '/mnt-gluster/data/yxue/research/allstate/out/mm/ITTITITTTV_sample_weights1_8bs_6005it_dd2.log'
	# sample_weights_fn = '/mnt-gluster/data/yxue/research/allstate/data/synthetic_mm_cifar_imdb_hmdb/sample_weights_ITTITITTTV_1.json'

	print(weighted_acc(args.log_fn, args.sample_weights_fn))