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

OmniNet evalution script.

"""
import argparse
import os
import torch
import pickle
import time
import json
import numpy as np
import glob
import libs.omninet as omninet
from libs.utils import dataloaders as dl
from tensorboardX import SummaryWriter
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import libs.omninet.routines as r
from libs.omninet.util import ScheduledOptim
from torch.optim.adam import Adam
import random
from nltk.tokenize import word_tokenize
import sys
from tqdm import tqdm
from PIL import Image
from libs.utils.train_util import *
from libs.utils.cocoapi.coco import COCO
from libs.utils.vqa.vqa import VQA
from libs.utils.vqa.vqaEval import VQAEval
from libs.utils.cocoapi.eval import COCOEvalCap
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from libs.utils.bleu import compute_bleu

from omninet_config import *

# coco_images = '/data/coco/train_val'
# caption_dir = '/data/coco'
# vqa_dir = '/data/vqa'
# hmdb_data_dir='/data/hmdb'
# hmdb_process_dir='/data/hmdbprocess'
# model_save_path = 'checkpoints'

data_path = '/files/yxue/research/allstate/data'

coco_images = os.path.join(data_path, 'coco/train_val')
caption_dir = os.path.join(data_path, 'coco')
vqa_dir = os.path.join(data_path, 'vqa')


dropout = 0.5
image_height = 224
image_width = 224


class ImageDataset(Dataset):
    def __init__(self, img_list,transform):
        self.img_list=img_list
        self.N=len(self.img_list)
        self.transform=transform
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img).convert("RGB")
        return img
    

def in_vocab(answer, ans_to_id):
    for ans in answer[0]['answers']:
        if ans['answer'] in ans_to_id:
            return True
    return False

def vqa_evaluate_skipping(model,batch_size,out_fn='vqa_prediction.json',structured_path=None,log_fn='out/log'):
    predictions=[]
    vqa_val_ques=os.path.join(vqa_dir,'v2_OpenEnded_mscoco_val2014_questions.json')
    vqa_val_ann=os.path.join(vqa_dir,'v2_mscoco_val2014_annotations.json')
    vocab_file=os.path.join('conf/vqa_vocab.pkl')
    with open(vocab_file,'rb') as f:
            ans_to_id,id_to_ans=pickle.loads(f.read())
    vqa = VQA(annotation_file=vqa_val_ann, question_file=vqa_val_ques)
    questions=[]
    img_list=[]
    ques_ids=[]
    answers=[]
    structured = None
    #Multiple choice answer evaluation
    
    if structured_path is not None:
        with open(os.path.join(structured_path, 'synthetic_val_structured_data.dict'), 'rb') as f:
            val_structured = pickle.load(f)
        structured = []

    for i,q in enumerate(vqa.questions['questions']):
        answer = vqa.loadQA(q['question_id'])
        m_a=answer[0]['multiple_choice_answer']
        # if m_a in ans_to_id:
        if in_vocab(answer, ans_to_id):
            img_list.append(os.path.join(coco_images, '%012d.jpg' % (q['image_id'])))
            questions.append(q['question'])
            ques_ids.append(q['question_id'])
            # answer = vqa.loadQA(q['question_id'])
            answers.append(answer[0]['multiple_choice_answer'])

            if structured_path is not None:
                structured.append(val_structured[i])
    answers=np.array(answers)

    if not os.path.exists(out_fn):
        #Validation transformations
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        val_tfms = transforms.Compose([
                                            transforms.Resize(int(224*1.14)),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            normalize,
                                        ])
        
        val_dataset = ImageDataset(img_list,val_tfms)
        val_dl= DataLoader(val_dataset, num_workers=0, batch_size=batch_size,
                                         drop_last=False,pin_memory=True,shuffle=False)
        counter=0
        result_json=[]
        struct=None
        
        for b in tqdm(val_dl):
                imgs = b.cuda(0)
                ques=questions[counter:counter+imgs.shape[0]]
                if structured_path is not None:
                    struct = torch.tensor(structured[counter:counter+imgs.shape[0]]).cuda(0)
                preds,_,_ = r.vqa(model, imgs, ques,structured=struct, mode='predict',return_str_preds=True,num_steps=1)
                preds=preds.reshape([-1]).cpu().numpy()
                for p in preds:
                    result_json.append({'question_id':ques_ids[counter],'answer':id_to_ans[p]})
                    counter+=1
        with open(out_fn, 'w') as outfile:
            json.dump(result_json, outfile)

    print('loading predictions..')
    #Evaluate the predictions 
    predictions=[]
    vqaRes=vqa.loadRes(out_fn,vqa_val_ques,ques_ids=ques_ids)
    vqaEval = VQAEval(vqa, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2
    with open(out_fn, 'r') as f:
        json_ans = json.load(f)
    for j in json_ans:
        predictions.append(j['answer'])
    predictions=np.array(predictions)
    print(np.sum(predictions==answers)/predictions.shape[0])
    log_str = '{}\n'.format(np.sum(predictions==answers)/predictions.shape[0])
    # evaluate results
    """
    If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
    By default it uses all the question ids in annotation file
    """
    vqaEval.evaluate(ques_ids)
    print('\n\nVQA Evaluation results')
    print('-'*50)

    log_str += '\n\nVQA Evaluation results\n' + '-'*50 + '\n\n'

    # print accuracies
    print("\n")
    print("Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall']))
    print("Per Question Type Accuracy is the following:")

    log_str += "Overall Accuracy is: %.02f\n\nPer Question Type Accuracy is the following:\n" %(vqaEval.accuracy['overall'])

    for quesType in vqaEval.accuracy['perQuestionType']:
        print("%s : %.02f" %(quesType, vqaEval.accuracy['perQuestionType'][quesType]))
        log_str += "%s : %.02f\n" %(quesType, vqaEval.accuracy['perQuestionType'][quesType])
    log_str += '\nPer Answer Type Accuracy is the following:\n'
    print("\n")
    print("Per Answer Type Accuracy is the following:")
    for ansType in vqaEval.accuracy['perAnswerType']:
        print("%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType]))
        log_str += "%s : %.02f\n" %(ansType, vqaEval.accuracy['perAnswerType'][ansType])
    log_str += '\n'
    print("\n")

    with open(log_fn, 'w') as f:
        print(log_str, file=f)
        

def vqa_evaluate(model,batch_size):
    predictions=[]
    vqa_val_ques=os.path.join(vqa_dir,'v2_OpenEnded_mscoco_val2014_questions.json')
    vqa_val_ann=os.path.join(vqa_dir,'v2_mscoco_val2014_annotations.json')
    vocab_file=os.path.join('conf/vqa_vocab.pkl')
    with open(vocab_file,'rb') as f:
            ans_to_id,id_to_ans=pickle.loads(f.read())
    vqa = VQA(annotation_file=vqa_val_ann, question_file=vqa_val_ques)
    questions=[]
    img_list=[]
    ques_ids=[]
    answers=[]
    #Multiple choice answer evaluation
    
    for q in vqa.questions['questions']:
        img_list.append(os.path.join(coco_images, '%012d.jpg' % (q['image_id'])))
        questions.append(q['question'])
        ques_ids.append(q['question_id'])
        answer = vqa.loadQA(q['question_id'])
        answers.append(answer[0]['multiple_choice_answer'])
    answers=np.array(answers)
    #Validation transformations
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_tfms = transforms.Compose([
                                        transforms.Resize(int(224*1.14)),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize,
                                    ])
    
    val_dataset = ImageDataset(img_list,val_tfms)
    val_dl= DataLoader(val_dataset, num_workers=0, batch_size=batch_size,
                                     drop_last=False,pin_memory=True,shuffle=False)
    counter=0
    result_json=[]
    
    for b in tqdm(val_dl):
            imgs = b.cuda(0)
            ques=questions[counter:counter+imgs.shape[0]]
            preds,_,_ = r.vqa(model, imgs, ques, mode='predict',return_str_preds=True,num_steps=1)
            preds=preds.reshape([-1]).cpu().numpy()
            for p in preds:
                result_json.append({'question_id':ques_ids[counter],'answer':id_to_ans[p]})
                counter+=1
    with open('results/vqa_prediction.json', 'w') as outfile:
        json.dump(result_json, outfile)
    #Evaluate the predictions 
    predictions=[]
    vqaRes=vqa.loadRes('results/vqa_prediction.json',vqa_val_ques)
    vqaEval = VQAEval(vqa, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2
    with open('results/vqa_prediction.json', 'r') as f:
        json_ans = json.load(f)
    for j in json_ans:
        predictions.append(j['answer'])
    predictions=np.array(predictions)
    print(np.sum(predictions==answers)/predictions.shape[0])
    # evaluate results
    """
    If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
    By default it uses all the question ids in annotation file
    """
    vqaEval.evaluate()
    print('\n\nVQA Evaluation results')
    print('-'*50)

    # print accuracies
    print("\n")
    print("Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall']))
    print("Per Question Type Accuracy is the following:")
    for quesType in vqaEval.accuracy['perQuestionType']:
        print("%s : %.02f" %(quesType, vqaEval.accuracy['perQuestionType'][quesType]))
    print("\n")
    print("Per Answer Type Accuracy is the following:")
    for ansType in vqaEval.accuracy['perAnswerType']:
        print("%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType]))
    print("\n")

def vqa_struct_evaluate(model,batch_size,structured_path,out_fn='vqa_prediction.json',fusion=True):
    predictions=[]
    vqa_val_ques=os.path.join(vqa_dir,'v2_OpenEnded_mscoco_val2014_questions.json')
    vqa_val_ann=os.path.join(vqa_dir,'v2_mscoco_val2014_annotations.json')
    vocab_file=os.path.join('conf/vqa_vocab.pkl')
    with open(vocab_file,'rb') as f:
            ans_to_id,id_to_ans=pickle.loads(f.read())
    vqa = VQA(annotation_file=vqa_val_ann, question_file=vqa_val_ques)
    questions=[]
    img_list=[]
    ques_ids=[]
    answers=[]
    #Multiple choice answer evaluation
    structured=[]

    with open(os.path.join(structured_path, 'synthetic_structured_data_normed.dict'), 'rb') as f:
        synthetic_structured_data = pickle.load(f)
    val_structured = synthetic_structured_data['val']
    val_ids = list(val_structured.keys())

    n_in_vocab = 0
    n_out_vocab = 0
    for q in vqa.questions['questions']:
        img_list.append(os.path.join(coco_images, '%012d.jpg' % (q['image_id'])))
        questions.append(q['question'])
        ques_ids.append(q['question_id'])
        answer = vqa.loadQA(q['question_id'])
        answers.append(answer[0]['multiple_choice_answer'])
        if q['question_id'] in val_structured:
            structured.append(val_structured[q['question_id']])
            n_in_vocab += 1
        else:
            structured.append(val_structured[np.random.choice(val_ids, 1)[0]])
            n_out_vocab += 1

    print('n_in_vocab:', n_in_vocab)
    print('n_out_vocab:', n_out_vocab)
    print('n_struct:', len(val_structured))

    answers=np.array(answers)
    if not os.path.exists(out_fn):
        #Validation transformations
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        val_tfms = transforms.Compose([
                                            transforms.Resize(int(224*1.14)),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            normalize,
                                        ])
        
        val_dataset = ImageDataset(img_list,val_tfms)
        val_dl= DataLoader(val_dataset, num_workers=2, batch_size=batch_size,
                                         drop_last=False,pin_memory=True,shuffle=False)
        counter=0
        result_json=[]
        
        for b in tqdm(val_dl):
                imgs = b.cuda(0)
                ques=questions[counter:counter+imgs.shape[0]]
                struct = torch.as_tensor(structured[counter:counter+imgs.shape[0]]).float().cuda(0)
                if fusion:
                    preds = r.vqa_fusion(model, imgs, ques,structured=struct, mode='predict',return_str_preds=True,num_steps=1)[0]
                else:
                    preds = r.vqa(model, imgs, ques,structured=struct, mode='predict',return_str_preds=True,num_steps=1)[0]
                preds=preds.reshape([-1]).cpu().numpy()
                for p in preds:
                    result_json.append({'question_id':ques_ids[counter],'answer':id_to_ans[p]})
                    counter+=1
        with open(out_fn, 'w') as outfile:
            json.dump(result_json, outfile)
    #Evaluate the predictions 
    predictions=[]
    vqaRes=vqa.loadRes(out_fn,vqa_val_ques)
    vqaEval = VQAEval(vqa, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2
    with open(out_fn, 'r') as f:
        json_ans = json.load(f)
    for j in json_ans:
        predictions.append(j['answer'])
    predictions=np.array(predictions)
    print(np.sum(predictions==answers)/predictions.shape[0])
    # evaluate results
    """
    If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
    By default it uses all the question ids in annotation file
    """
    vqaEval.evaluate()
    print('\n\nVQA Evaluation results')
    print('-'*50)

    # print accuracies
    print("\n")
    print("Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall']))
    print("Per Question Type Accuracy is the following:")
    for quesType in vqaEval.accuracy['perQuestionType']:
        print("%s : %.02f" %(quesType, vqaEval.accuracy['perQuestionType'][quesType]))
    print("\n")
    print("Per Answer Type Accuracy is the following:")
    for ansType in vqaEval.accuracy['perAnswerType']:
        print("%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType]))
    print("\n")

# def vqa_evaluate(model,batch_size,out_fn='vqa_prediction.json',structured_path=None,log_fn='out/log'):
#     predictions=[]
#     vqa_val_ques=os.path.join(vqa_dir,'v2_OpenEnded_mscoco_val2014_questions.json')
#     vqa_val_ann=os.path.join(vqa_dir,'v2_mscoco_val2014_annotations.json')
#     vocab_file=os.path.join('conf/vqa_vocab.pkl')
#     with open(vocab_file,'rb') as f:
#             ans_to_id,id_to_ans=pickle.loads(f.read())
#     vqa = VQA(annotation_file=vqa_val_ann, question_file=vqa_val_ques)
#     questions=[]
#     img_list=[]
#     ques_ids=[]
#     answers=[]
#     structured = None
#     #Multiple choice answer evaluation
    
#     if structured_path is not None:
#         with open(os.path.join(structured_path, 'synthetic_val_structured_data.dict'), 'rb') as f:
#             val_structured = pickle.load(f)
#         structured = []

#     for i,q in enumerate(vqa.questions['questions']):
#         img_list.append(os.path.join(coco_images, '%012d.jpg' % (q['image_id'])))
#         questions.append(q['question'])
#         ques_ids.append(q['question_id'])
#         answer = vqa.loadQA(q['question_id'])
#         answers.append(answer[0]['multiple_choice_answer'])
#         if structured_path is not None:
#             if i in val_structured:
#                 structured.append(val_structured[i])
#             else:
#                 structured.append(np.random.random(512))

#     answers=np.array(answers)

#     if not os.path.exists(out_fn):
#         #Validation transformations
#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         val_tfms = transforms.Compose([
#                                             transforms.Resize(int(224*1.14)),
#                                             transforms.CenterCrop(224),
#                                             transforms.ToTensor(),
#                                             normalize,
#                                         ])
        
#         val_dataset = ImageDataset(img_list,val_tfms)
#         val_dl= DataLoader(val_dataset, num_workers=0, batch_size=batch_size,
#                                          drop_last=False,pin_memory=True,shuffle=False)
#         counter=0
#         result_json=[]
#         struct=None
        
#         for b in tqdm(val_dl):
#             imgs = b.cuda(0)
#             ques=questions[counter:counter+imgs.shape[0]]
#             if structured_path is not None:
#                 print([x.shape() for x in structured[counter:counter+imgs.shape[0]]])
#                 struct = torch.tensor(structured[counter:counter+imgs.shape[0]]).float().cuda(0)
#             preds,_,_ = r.vqa(model, imgs, ques,structured=struct, mode='predict',return_str_preds=True,num_steps=1)
#             preds=preds.reshape([-1]).cpu().numpy()
#             for p in preds:
#                 result_json.append({'question_id':ques_ids[counter],'answer':id_to_ans[p]})
#                 counter+=1
#         with open(out_fn, 'w') as outfile:
#             json.dump(result_json, outfile)

#     print('loading predictions..')
#     #Evaluate the predictions 
#     predictions=[]
#     vqaRes=vqa.loadRes(out_fn,vqa_val_ques)
#     vqaEval = VQAEval(vqa, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2
#     with open(out_fn, 'r') as f:
#         json_ans = json.load(f)
#     for j in json_ans:
#         predictions.append(j['answer'])
#     predictions=np.array(predictions)
#     print(np.sum(predictions==answers)/predictions.shape[0])
#     log_str = '{}\n'.format(np.sum(predictions==answers)/predictions.shape[0])
#     # evaluate results
#     """
#     If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
#     By default it uses all the question ids in annotation file
#     """
#     vqaEval.evaluate()
#     print('\n\nVQA Evaluation results')
#     print('-'*50)

#     log_str += '\n\nVQA Evaluation results\n' + '-'*50 + '\n\n'

#     # print accuracies
#     print("\n")
#     print("Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall']))
#     print("Per Question Type Accuracy is the following:")

#     log_str += "Overall Accuracy is: %.02f\n\nPer Question Type Accuracy is the following:\n" %(vqaEval.accuracy['overall'])

#     for quesType in vqaEval.accuracy['perQuestionType']:
#         print("%s : %.02f" %(quesType, vqaEval.accuracy['perQuestionType'][quesType]))
#         log_str += "%s : %.02f\n" %(quesType, vqaEval.accuracy['perQuestionType'][quesType])
#     log_str += '\nPer Answer Type Accuracy is the following:\n'
#     print("\n")
#     print("Per Answer Type Accuracy is the following:")
#     for ansType in vqaEval.accuracy['perAnswerType']:
#         print("%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType]))
#         log_str += "%s : %.02f\n" %(ansType, vqaEval.accuracy['perAnswerType'][ansType])
#     log_str += '\n'
#     print("\n")

#     with open(log_fn, 'w') as f:
#         print(log_str, file=f)
        

def coco_evaluate(model,batch_size):
    predictions=[]
    val_ann_file=os.path.join(caption_dir,'captions_val2017.json')
    coco = COCO(val_ann_file)
    img_ids=coco.getImgIds()
    img_list=[]
    for i in img_ids:
        img_list.append(os.path.join(coco_images, '%012d.jpg' % (i)))
    #Validation transformations
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_tfms = transforms.Compose([
                                        transforms.Resize(int(224*1.14)),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize,
                                    ])
    
    val_dataset = ImageDataset(img_list,val_tfms)
    val_dl= DataLoader(val_dataset, num_workers=0, batch_size=batch_size,
                                     drop_last=False,pin_memory=True,shuffle=False)
    counter=0
    result_json=[]
    for b in tqdm(val_dl):
            imgs = b.cuda(0)
            preds,_,_ = r.image_caption(model, imgs, mode='predict',return_str_preds=True)
            for p in preds:
                result_json.append({'image_id':img_ids[counter],'caption':p})
                counter+=1
                
    with open('results/caption_prediction.json', 'w') as outfile:
        json.dump(result_json, outfile)
    #Evaluate the predictions 
    cocoRes=coco.loadRes('results/caption_prediction.json')
    cocoEval = COCOEvalCap(coco, cocoRes)
    
    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    cocoEval.evaluate()
    print('\n\nCOCO Evaluation results')
    print('-'*50)
    for metric, score in cocoEval.eval.items():
        print('%s: %.3f'%(metric, score))
                
def hmdb_evaluate(model,batch_size):
    _,test_dl=dl.hmdb_batchgen(hmdb_data_dir,hmdb_process_dir,num_workers=0,batch_size=batch_size,
                                         test_batch_size=batch_size,
                                           clip_len=16)
    correct=0
    total=0
    for b in tqdm(test_dl):
            vid,labels = b
            vid = vid.cuda(device=0)
            preds,_,_ = r.hmdb(model, vid,mode='predict',return_str_preds=True,num_steps=1)
            preds=preds.reshape([-1]).cpu().numpy()
            labels=labels.reshape([-1]).cpu().numpy()
            correct+=np.sum(preds==labels)
            total+=labels.shape[0]
    accuracy=(correct/total)*100
    print('\n\nHMDB Evaluation results')
    print('-'*50)
    print('Accuracy: %f%%'%(accuracy))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OmniNet Evaluation script.')
    parser.add_argument('task', help='Task for which to evaluate for.')
    parser.add_argument('model_file', help='Model file to evaluate with.')
    parser.add_argument('--batch_size', default=32, help='Batch size')
    parser.add_argument('--structured_path', default=None, help='path to structured data.')
    parser.add_argument('--out_fn', help='filename to save results.')
    parser.add_argument('--log_fn', help='filename to save logs.')
    parser.add_argument('--skipping', action='store_true', help='true if skip questions as in training.')
    parser.add_argument('--inject_at_logits', action='store_true', help='True if inject structured feature at final logit layer.')
    parser.add_argument('--inject_at_encoder', action='store_true', help='True if inject structured feature at encoder.')
    parser.add_argument('--inject_after_encoder', action='store_true', help='True if inject structured feature right after encoder.')
    parser.add_argument('--inject_at_decoder', action='store_true', help='True if inject structured feature at decoder.')
    parser.add_argument('--temp_fusion_attn_type', default='default', type=str, help='Attention type in fusion.')
    parser.add_argument('--spat_fusion_attn_type', default='default', type=str, help='Attention type in fusion.')
    parser.add_argument('--convex_gate', action='store_true', help='True if use convex gate.')
    parser.add_argument('--pooling', action='store_true', help='True if use the whole image feature vector for gating.')
    parser.add_argument('--fusion', action='store_true', help='True if fuse structured features with unstructured features by attention-based fusion.')
    parser.add_argument('--struct_periph_dropout', default=0.1, type=float, help='dropout rates at struct peripherals')
    parser.add_argument('--struct_temp_periph_dropout', default=0.1, type=float, help='dropout rates at struct temp peripherals')
    parser.add_argument('--struct_spat_periph_dropout', default=0.1, type=float, help='dropout rates at struct spat peripherals')

    args = parser.parse_args()
    torch.manual_seed(47)
    np.random.seed(47)
    task=args.task
    batch_size=int(args.batch_size)
    model_file=args.model_file

    if not os.path.exists(args.out_fn):
        config = vqa_struct_config()
        config[0]['temp_fusion_attn_type'] = args.temp_fusion_attn_type
        config[0]['spat_fusion_attn_type'] = args.spat_fusion_attn_type
        config[0]['inject_at_logits'] = args.inject_at_logits
        config[0]['inject_at_encoder'] = args.inject_at_encoder
        config[0]['inject_after_encoder'] = args.inject_after_encoder
        config[0]['convex_gate'] = args.convex_gate
        config[0]['pooling'] = args.pooling
        config[1]['struct_dropout'] = args.struct_periph_dropout
        config[1]['struct_temp_dropout'] = args.struct_temp_periph_dropout
        config[1]['struct_spat_dropout'] = args.struct_spat_periph_dropout
        if args.inject_at_decoder:
            config[0]['inject_at_decoder'] = True
            config[0]['use_s_decoder'] = True

        model = omninet.OmniNet(gpu_id=0, config=config)
        model.restore_file(model_file)
       
        model=model.to(0)
        model=model.eval()
    else:
        model = None

    if task=='caption':
        coco_evaluate(model,batch_size)
    elif task=='vqa' or 'vqa_struct':
        if args.skipping:
            vqa_evaluate_skipping(model,batch_size,out_fn=args.out_fn,structured_path=args.structured_path,log_fn=args.log_fn)
        else:
            if task == 'vqa':
                vqa_evaluate(model,batch_size,out_fn=args.out_fn,log_fn=args.log_fn)
            else:
                vqa_struct_evaluate(model,batch_size,args.structured_path,out_fn=args.out_fn,fusion=args.fusion)

    elif task=='hmdb':
        hmdb_evaluate(model, batch_size)
    else:
        print('Invalid task provided')
    