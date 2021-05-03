import numpy as np
import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
from libs.utils.vqa.vqa import VQA
import tqdm

vqa_dir = '/files/yxue/research/allstate/data/vqa'

# cls_sizes_tr_fn = '/files/yxue/research/allstate/data/vqa/class_sizes_tr'
# cls_sizes_val_fn = '/files/yxue/research/allstate/data/vqa/class_sizes_val'

cls_sizes_tr_fn = '/files/yxue/research/allstate/data/vqa/class_sizes_tr_in_vocab'
cls_sizes_val_fn = '/files/yxue/research/allstate/data/vqa/class_sizes_val_in_vocab'


def ans_in_vocab(answer, ans_to_id):
    for ans in answer[0]['answers']:
        if ans['answer'] in ans_to_id:
            return ans['answer']
    return None

def get_class_sizes():
    vqa_train_ques=os.path.join(vqa_dir,'v2_OpenEnded_mscoco_train2014_questions.json')
    vqa_train_ann=os.path.join(vqa_dir,'v2_mscoco_train2014_annotations.json')
    vqa_val_ques=os.path.join(vqa_dir,'v2_OpenEnded_mscoco_val2014_questions.json')
    vqa_val_ann=os.path.join(vqa_dir,'v2_mscoco_val2014_annotations.json')
    
    vocab_file=os.path.join('conf/vqa_vocab.pkl')
    
    vqa = VQA(annotation_file=vqa_train_ann, question_file=vqa_train_ques)

    val_vqa = VQA(annotation_file=vqa_val_ann, question_file=vqa_val_ques)
    
    with open(vocab_file,'rb') as f:
        ans_to_id,id_to_ans=pickle.loads(f.read())
    
    ans = []
    ques = vqa.questions['questions']
    for i,x in tqdm.tqdm(enumerate(ques),'Loading VQA data to memory'):
        answer = vqa.loadQA(x['question_id'])
        m_a=answer[0]['multiple_choice_answer']
        # if m_a in ans_to_id:
        #         ans.append(ans_to_id[m_a])

        a = ans_in_vocab(answer, ans_to_id)
        if m_a in ans_to_id:
            ans.append(ans_to_id[m_a])
        elif a is not None:
            ans.append(ans_to_id[a])

    val_ans = []
    val_ques = val_vqa.questions['questions']
    for i,x in tqdm.tqdm(enumerate(val_ques),'Loading VQA data to memory'):
        answer = val_vqa.loadQA(x['question_id'])
        m_a=answer[0]['multiple_choice_answer']
        # if m_a in ans_to_id:
        #         val_ans.append(ans_to_id[m_a])

        a = ans_in_vocab(answer, ans_to_id)
        if m_a in ans_to_id:
            val_ans.append(ans_to_id[m_a])
        elif a is not None:
            val_ans.append(ans_to_id[a])

    
    ids, cnts = np.unique(ans, return_counts=True)
    val_ids, val_cnts = np.unique(val_ans, return_counts=True)
    with open(cls_sizes_tr_fn, 'wb') as f:
        pickle.dump(dict(zip(ids,cnts)),f)
    
    with open(cls_sizes_val_fn, 'wb') as f:
        pickle.dump(dict(zip(val_ids,val_cnts)),f)
    

def gen_structured_data(feature_size, n_cls, std, out_path):
    np.random.seed(812)

    with open(cls_sizes_tr_fn, 'rb') as f:
        cls_sizes_tr = pickle.load(f)

    with open(cls_sizes_val_fn, 'rb') as f:
        cls_sizes_val = pickle.load(f)

    centers = np.random.uniform(low=-1,high=1,size=(n_cls,feature_size))

    tr_data = {}
    val_data = {}

    for c in cls_sizes_tr:
        tr_data[c] = np.random.normal(centers[c], [std]*feature_size, (cls_sizes_tr[c],feature_size))

    for c in cls_sizes_val:
        val_data[c] = np.random.normal(centers[c], [std]*feature_size, (cls_sizes_val[c],feature_size))

    # train_X = np.vstack(list(tr_data.values()))
    # val_X = np.vstack(list(val_data.values()))

    # train_Y = np.hstack([[c]*cls_sizes_tr[c] for c in cls_sizes_tr])
    # val_Y = np.hstack([[c]*cls_sizes_val[c] for c in cls_sizes_val])

    """
    traing a logistic regression model on synthetic data


    class LogisticRegression(torch.nn.Module):
        def __init__(self, input_dim, output_dim):
            super(LogisticRegression, self).__init__()
            self.linear = torch.nn.Linear(input_dim, output_dim)

        def forward(self, x):
            outputs = self.linear(x)
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


    batch_size = 128
    epochs = 20
    lr_rate = 0.001

    dataset = StructuredDataset(train_X, train_Y)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0)
    val_dataset = StructuredDataset(val_X, val_Y)
    val_dl = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0)

    model = LogisticRegression(feature_size, n_cls)
    criterion = torch.nn.CrossEntropyLoss() # computes softmax and then the cross entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

    it = 0
    for epoch in range(int(epochs)):
        for i, b in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(b['x'])
            loss = criterion(outputs, b['y'])
            loss.backward()
            optimizer.step()
            
            it += 1

            if it%500 == 0:
                selected_val_mini_batch_ids = np.random.choice(range(len(val_dl)), 200, replace=False)

                correct = 0
                total = 0
                for bi, val_b in enumerate(val_dl):
                    if bi not in selected_val_mini_batch_ids:
                        continue
                    outputs = model(val_b['x'])
                    predicted = outputs.argmax(dim=1)
                    total+= val_b['y'].size(0)
                    # for gpu, bring the predicted and labels back to cpu fro python operations to work
                    correct+= (predicted == val_b['y']).sum()
                accuracy = 100 * correct/total
                print("epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), accuracy))

    """

    # with open('vqa/synthetic_structured_clustering_std1.7_valAcc78.pkl', 'wb') as f:
    #     pickle.dump({'train': tr_data, 'val': val_data}, f)

    with open(os.path.join(out_path, 'synthetic_structured_clustering_std%s.pkl'%std), 'wb') as f:
        pickle.dump({'train': tr_data, 'val': val_data}, f)


feature_size = 128
n_cls = 3500
std = 3
out_path = '/files/yxue/research/allstate/data/vqa'
# get_class_sizes()
# print('got class sizes')
gen_structured_data(feature_size, n_cls, std, out_path)
# print('got structured data')

def assign_structured_samples(ann_file, ques_file, structured):
    vqa = VQA(annotation_file=ann_file, question_file=ques_file)
    
    structured_dict = {}
    ans_dict = {}

    with open(vocab_file,'rb') as f:
        ans_to_id,id_to_ans=pickle.loads(f.read())
    # load the questions
    ques = vqa.questions['questions']
    # for tracking progress
    # for every question

    structured_idx = dict(zip(id_to_ans.keys(), [0]*len(id_to_ans)))

    j = 0
    k = 0
    for i,x in tqdm.tqdm(enumerate(ques),'Loading VQA data to memory'):
        # get the path
        answer = vqa.loadQA(x['question_id'])
        m_a=answer[0]['multiple_choice_answer']
        a = ans_in_vocab(answer, ans_to_id)
        if m_a in ans_to_id:
            ans_id = ans_to_id[m_a]
            j += 1
        elif a is not None:
            ans_id = ans_to_id[a]
            k += 1
        else:
            ans_id = None

        if ans_id is not None:
            structured_dict[x['question_id']] = structured[ans_id][structured_idx[ans_id]]
            structured_idx[ans_id] += 1
            ans_dict[x['question_id']] = ans_id

    # print('num of ques:', i+1)
    # print('num of ma:', j)
    # print('num of in_vocab:', k)
    # print('num of structured:', sum(structured_idx.values()))
    # print('len of structured_dict:', len(structured_dict))
    return structured_dict, ans_dict

def normalizing(structured_path):
    with open(os.path.join(structured_path, 'synthetic_structured_data.dict'), 'rb') as f:
        d = pickle.load(f)

    tr, val = d['train'], d['val']
    all_tr = np.vstack(list(tr.values()))
    mn = all_tr.mean(axis=0)
    std = all_tr.std(axis=0)

    for k in tr:
        tr[k] = (tr[k] - mn)/std

    for k in val:
        val[k] = (val[k] - mn)/std

    with open(os.path.join(structured_path, 'synthetic_structured_data_normed.dict'), 'wb') as f:
        pickle.dump({'train': tr, 'val': val}, f)

vqa_train_ques=os.path.join(vqa_dir,'v2_OpenEnded_mscoco_train2014_questions.json')
vqa_train_ann=os.path.join(vqa_dir,'v2_mscoco_train2014_annotations.json')
vqa_val_ques=os.path.join(vqa_dir,'v2_OpenEnded_mscoco_val2014_questions.json')
vqa_val_ann=os.path.join(vqa_dir,'v2_mscoco_val2014_annotations.json')
vocab_file=os.path.join('conf/vqa_vocab.pkl')

with open(os.path.join(out_path, 'synthetic_structured_clustering_std%s.pkl'%std), 'rb') as f:
    structured = pickle.load(f)

structured_tr, label_tr = assign_structured_samples(vqa_train_ann, vqa_train_ques, structured['train'])
structured_val, label_val = assign_structured_samples(vqa_val_ann, vqa_val_ques, structured['val'])

structured_path = os.path.join(out_path, 'synthetic_structured_clustering_std%s'%std)
if not os.path.exists(structured_path):
    os.makedirs(structured_path)

with open(os.path.join(structured_path, 'synthetic_structured_data.dict'), 'wb') as f:
    pickle.dump({'train': structured_tr, 'val': structured_val}, f)

with open(os.path.join(structured_path, 'synthetic_structured_data_labels.dict'), 'wb') as f:
    pickle.dump({'train': label_tr, 'val': label_val}, f)

normalizing(structured_path)
