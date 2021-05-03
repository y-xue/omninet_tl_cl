import numpy as np
import random
import pickle

import torch
import torch.backends.cudnn as cudnn
from torch.nn.functional import log_softmax
from torch.utils.data import Dataset, DataLoader
from libs.omninet.util import ScheduledOptim

import os
import argparse

parser = argparse.ArgumentParser(description='train structured')
parser.add_argument('--out_path', 
    default='/files/yxue/research/allstate/out/omninet/structured', 
    type=str, help='output_path')

args = parser.parse_args()

np.random.seed(812)

batch_size = 64
epochs = 200
input_dim = 128
output_dim = 3500
# lr_rate = 0.02

if not os.path.exists(args.out_path):
    os.makedirs(args.out_path)

data_path='/files/yxue/research/allstate/data/vqa/synthetic_structured_clustering_std3'

class FFN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(FFN, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        outputs = log_softmax(x, dim=1)
        return outputs


class StructuredDataset(Dataset):
    def __init__(self, X, Y):
        
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'x': torch.tensor(self.X[idx]).float(), 'y': torch.tensor(self.Y[idx])}
        
        return sample

with open(data_path+'/synthetic_structured_data_labels.dict', 'rb') as f:
    Y = pickle.load(f)

with open(data_path+'/synthetic_structured_data_normed.dict', 'rb') as f:
    X = pickle.load(f)

train_Y, val_Y = np.hstack(list(Y['train'].values())), np.hstack(list(Y['val'].values()))
train_X, val_X = np.vstack(list(X['train'].values())), np.vstack(list(X['val'].values()))

dataset = StructuredDataset(train_X, train_Y)
dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)
val_dataset = StructuredDataset(val_X, val_Y)
val_dl = DataLoader(val_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)

print(len(dataloader))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = FFN(input_dim, output_dim, hidden_dim=512)
# criterion = torch.nn.CrossEntropyLoss() # computes softmax and then the cross entropy
# optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
criterion = torch.nn.NLLLoss()
optimizer = ScheduledOptim(
            torch.optim.Adam(
                filter(lambda x: x.requires_grad, model.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            512, 16000,0,max_lr=0.0001,init_lr=0.02)

model = model.to(device)
if device == 'cuda':
    # model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

print(sum(p.numel() for p in model.parameters() if p.requires_grad))

def train(model, dataloader, val_dl):
    best_acc = 0

    model = model.train()
    it = -1
    log_str = ''
    for epoch in range(int(epochs)):
        for i, b in enumerate(dataloader):
            it += 1
            optimizer.zero_grad()

            x = b['x'].to(device)
            y = b['y'].to(device)
            outputs = model(x)
            predicted = torch.reshape(outputs.argmax(dim=1), [-1]).float()
            
            loss = criterion(outputs, y)
            n_correct = (predicted == y.float()).sum().cpu().numpy()
            n_total = y.size(0)
            acc = 100 * (n_correct / n_total)
            loss.backward()
            optimizer.step()

            print('Step %d, STRUCT Loss: %f, Accuracy:  %f %%' % (it, loss.detach(),acc))
            log_str += 'Step %d, STRUCT Loss: %f, Accuracy:  %f %%\n' % (it, loss.detach(),acc)
        
        model = model.eval()
        # selected_val_mini_batch_ids = np.random.choice(range(len(val_dl)), 200, replace=False)
        correct = 0
        total = 0
        for bi, val_b in enumerate(val_dl):
            # if bi not in selected_val_mini_batch_ids:
            #     continue
            x = val_b['x'].to(device)
            y = val_b['y'].to(device)

            outputs = model(x)
            predicted = torch.reshape(outputs.argmax(dim=1), [-1]).float()

            loss = criterion(outputs, y)
            total += y.size(0)
            # for gpu, bring the predicted and labels back to cpu fro python operations to work
            correct+= (predicted == y.float()).sum().cpu().numpy()
        accuracy = 100 * correct/total

        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), os.path.join(args.out_path, 'best_model.pth'))
            log_str += 'best_iter: %s, saving model at epoch %s\n'%(it, epoch)

        # print("epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), accuracy))
        print('Step %d, STRUCT validation loss: %f, Accuracy %f %%' % (it, float(loss.detach().cpu().numpy()), accuracy))
        log_str += 'Step %d, STRUCT validation loss: %f, Accuracy %f %%\n' % (it, float(loss.detach().cpu().numpy()), accuracy)
        
        with open(args.out_path+'.log', 'a') as f:
            print(log_str, file=f)
            log_str = ''

        model = model.train()

train(model, dataloader, val_dl)
