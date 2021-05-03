import matplotlib.pyplot as plt
import re
import numpy as np
import os

def get_val_acc(log):
    val_acc_steps = re.findall(r'Step (\d+),.*Accuracy (\d+\.\d+) \%', log)
    val_acc = [float(x[1]) for x in val_acc_steps]
    val_steps = [int(x[0]) for x in val_acc_steps]
    
    return val_acc, val_steps

def get_tr_acc(log):
    tr_acc = re.findall(r'.*Accuracy:  (\d+\.\d+) \%', log)
    tr_acc = [float(x) for x in tr_acc]
    
    return tr_acc


#"Step 122909, VQA Loss: 1.832767, Spat Emb Loss: 0.002828, Temp Emb Loss: 0.004692, Accuracy:  51.562500 %"
def get_tr_emb_loss(log):
    tr_loss = re.findall(r'.*VQA Loss: (\d+\.\d+), Spat Emb Loss: (\d+\.\d+),'
                         ' Temp Emb Loss: (\d+\.\d+), Accuracy.*\%', log)
    
    pred_loss = np.array([float(x[0]) for x in tr_loss])
    spat_emb_loss = np.array([float(x[1]) for x in tr_loss])
    temp_emb_loss = np.array([float(x[2]) for x in tr_loss])
    
    return pred_loss, spat_emb_loss, temp_emb_loss

# 'Step 122912, VQA validation Loss: 2.684138, Spat Emb Loss: 0.002739, Temp Emb Loss: 0.003597, Accuracy 36.469790 %'
def get_val_emb_loss(log):
    loss = re.findall(r'.*VQA validation Loss: (\d+\.\d+), Spat Emb Loss: (\d+\.\d+),'
                         ' Temp Emb Loss: (\d+\.\d+), Accuracy.*\%', log)
    
    pred_loss = np.array([float(x[0]) for x in loss])
    spat_emb_loss = np.array([float(x[1]) for x in loss])
    temp_emb_loss = np.array([float(x[2]) for x in loss])
    
    return pred_loss, spat_emb_loss, temp_emb_loss

def get_val_loss(log):
    loss = re.findall(r'.*VQA validation loss: (\d+\.\d+),.*', log)
    return np.array([float(x) for x in loss])

def get_tr_loss(log):
    tr_loss = re.findall(r'.*VQA Loss: (\d+\.\d+).*\%', log)
    return np.array([float(x) for x in tr_loss])

def get_epoch_curve(x, n_steps, n_epochs):
    avg_x = []
    for i in range(n_epochs):
        avg_x.append(np.mean(x[i*n_steps:(i+1)*n_steps]))
    
    return avg_x

def get_tr_epoch_acc(log, n_steps):
    tr_acc = get_tr_acc(log)
    n_epochs = (len(tr_acc)+n_steps-1)//n_steps
    
    tr_epoch_acc = []
    for i in range(n_epochs):
        tr_epoch_acc.append(np.mean(tr_acc[i*n_steps:(i+1)*n_steps]))
    
    return tr_epoch_acc

def plot_tr_acc(log):
    vqa_tr_acc = get_tr_acc(log)
    plt.plot(range(len(vqa_tr_acc)), vqa_tr_acc, label='vqa training acc')
    plt.xlabel('iteration')
    plt.ylabel('acc')
    plt.legend()

def plot_epoch_tr_acc(log, n_steps, n_epochs):
    vqa_tr_epoch_acc = get_tr_epoch_acc(vqa_log, n_steps, n_epochs)
    print(len(vqa_tr_epoch_acc))
    plt.plot(range(len(vqa_tr_epoch_acc)), vqa_tr_epoch_acc, label='vqa training acc')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend()
    
def get_accs(log, n_steps=6470):
    tr_epoch_acc = get_tr_epoch_acc(log, n_steps=n_steps)
    val_res = get_val_acc(log)
    val_acc = val_res[0]
    val_steps = val_res[1]
    
    return tr_epoch_acc, val_acc, val_steps

def plot_accs(tr_epoch_acc, val_acc, val_steps, e, label_name, eval_freq=6469):
    e = min(e, len(val_steps))
    if eval_freq == 6469:
        x = (np.array(val_steps[:e])-1)/eval_freq
    else:
        x = np.array(val_steps[:e])/eval_freq
    plt.plot(x, val_acc[:e], marker='o', 
         label=label_name+' val acc')
    plt.plot(x, tr_epoch_acc[:e], 
         label=label_name+' training acc')

def plot_compare(logs, ylim=(65,80)):
    plt.figure(figsize=(10,7))
    for name in logs:
        with open(logs[name], 'r') as f:
            log = f.read() 
        
        if name == '(baseline) inject at logits':
            eval_interval = 6469
        else:
            eval_interval = 6470
            
        r = get_accs(log, eval_interval)
        
        print(name, max(r[1]))

        plot_accs(r[0], r[1], r[2], len(r[1]), name, eval_interval)

    plt.xlabel('epoch')
    plt.ylabel('acc')
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend()

