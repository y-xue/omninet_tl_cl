import argparse
import os
import torch
# import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='OmniNet training script.')
parser.add_argument('--out_path', default='/out/test', help='path to save the model.')
parser.add_argument('--model_name', default='model', help='model name.')
parser.add_argument('--start_epoch', type=int, default=40, help='start epoch')
parser.add_argument('--end_epoch', type=int, default=80, help='end epoch.')

args = parser.parse_args()

def diff(d1, d2):
	w_diff = 0
	for k in d1:
		w1, w2 = d1[k], d2[k]
		w_diff += torch.abs(w1-w2).sum().item()
	return w_diff

start_epoch = args.start_epoch
end_epoch = args.end_epoch
eval_interval = 6469

fn = os.path.join(args.out_path,'%d/model.pth')

d_last = torch.load(fn%int(start_epoch*eval_interval))
diffs = []
for i in range(start_epoch+1, end_epoch+1):
	d = torch.load(fn%int(i*eval_interval))
	diffs.append(diff(d,d_last))
	d_last = d

print(diffs)
# plt.figure(figsize=(10,5))
# plt.plot(range(start_epoch+1,end_epoch+1), l)
# plt.xticks(np.arange(start_epoch+1,end_epoch+1, step=5))
# plt.xlabel('epoch')
# plt.ylabel('difference')
# plt.title(model_name+' weights difference')
# plt.savefig(args.out_path+'_weights_diff.png')

