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

OmniNet routines for various tasks described in the paper.

"""

import torch
import torch.nn as nn
import numpy as np

def mm(omninet,images,text,videos,structured,targets=None,mode='train',return_str_preds=False,num_steps=1,sample_weights=None):
    # allstate multi-modal routine

    # images: [[image1, image2, ...], [image1, image2, ...], ...]  # (b,n,3,h,w)
    # text: [[text1, text2,...], [text1, text2, ...], ...]
    # videos: (b,n,f,3,h,w)
    batch_size = 0
    # print('mm...')
    # print('images shape:', images.shape)
    # if text is None:
    #     print('text length: None')
    # else:
    #     print('text length:', len(text[0]))
    # omninet.reset(batch_size)

    if images is not None:
        batch_size = images.shape[0]
    elif text is not None:
        batch_size = len(text)
    elif videos is not None:
        batch_size = videos.shape[0]

    if batch_size == 0:
        raise ValueError("Inputs batch of size 0")

    omninet.reset(batch_size)

    if structured is not None:
        omninet.encode_structured(structured)

    if images is not None:
        shape=images.shape
        if len(shape)==5:
            for i in range(shape[1]):
                # if mode == 'val':
                #     print(images[:,i,:,:,:].squeeze().shape)
                omninet.encode_images(images[:,i,:,:,:].squeeze(1)) #torch.index_select(images, 1, torch.tensor([i])))
        else:
            omninet.encode_images(images)

    if text is not None:
        for i in range(len(text[0])):
            omninet.encode_englishtexts([t[i] for t in text])

    if videos is not None:
        for i in range(videos.shape[1]):
            omninet.encode_videos(videos[:,i,:,:,:,:].squeeze(1))

    predictions = omninet.decode_greedy('mm', num_steps=num_steps)

    # print(predictions.shape)
    # targets = targets.unsqueeze(1)
    # print(targets.shape)
    # print(predictions)
    # print(targets)

    if targets is not None:
        loss, acc, n_correct, n_total = calc_nll_loss_and_acc(predictions,targets,sample_weights=sample_weights)
    else:
        loss,acc=None, None
    if return_str_preds:
        # Return predictions in detokenized string format
        predictions = predictions.argmax(-1)
    return predictions, loss, acc, n_correct, n_total

def mm_vqa(omninet,images,text,videos,structured,targets=None,mode='train',return_str_preds=False,num_steps=1,sample_weights=None):
    # allstate multi-modal routine

    # images: [[image1, image2, ...], [image1, image2, ...], ...]  # (b,n,3,h,w)
    # text: [[text1, text2,...], [text1, text2, ...], ...]
    # videos: (b,n,f,3,h,w)
    batch_size = 0
    # print('mm...')
    # print('images shape:', images.shape)
    # if text is None:
    #     print('text length: None')
    # else:
    #     print('text length:', len(text[0]))
    # omninet.reset(batch_size)

    if images is not None:
        batch_size = images.shape[0]
    elif text is not None:
        batch_size = len(text)
    elif videos is not None:
        batch_size = videos.shape[0]

    if batch_size == 0:
        raise ValueError("Inputs batch of size 0")

    omninet.reset(batch_size)

    if structured is not None:
        omninet.encode_structured(structured)

    if images is not None:
        shape=images.shape
        if len(shape)==5:
            for i in range(shape[1]):
                omninet.encode_images(images[:,i,:,:,:].squeeze()) #torch.index_select(images, 1, torch.tensor([i])))
        else:
            omninet.encode_images(images)

    if text is not None:
        for i in range(len(text[0])):
            omninet.encode_englishtexts([t[i] for t in text])

    if videos is not None:
        for i in range(videos.shape[1]):
            omninet.encode_videos(videos[:,i,:,:,:].squeeze())

    

    predictions = omninet.decode_greedy('mm_vqa', num_steps=num_steps)

    # print(predictions.shape)
    # targets = targets.unsqueeze(1)
    # print(targets.shape)
    # print(predictions)
    # print(targets)

    if targets is not None:
        loss, acc, n_correct, n_total  = calc_nll_loss_and_acc(predictions,targets)#,sample_weights=sample_weights)
    else:
        loss,acc=None, None
    if return_str_preds:
        # Return predictions in detokenized string format
        predictions = predictions.argmax(-1)
    return predictions, loss, acc

def birds(omninet,images,structured=None,targets=None,mode='train',return_str_preds=False,num_steps=1):
    batch_size = images.shape[0]
    # print('batch_size:',batch_size)
    omninet.reset(batch_size)
    omninet.encode_images(images,domain='IMAGE')
    # pad_mask = omninet.encode_englishtexts(text)
    if structured is not None:
        omninet.encode_structured(structured)#, pad_mask=pad_mask)
    # if mode in ['train','val']:
    #     predictions = omninet.decode_from_targets('birds', targets=targets)
    # elif mode=='predict':
    predictions = omninet.decode_greedy('birds', num_steps=num_steps)
    # Calculate loss if targets is provided
    # print(predictions.shape)
    # targets = targets.unsqueeze(1)
    # print(targets.shape)
    if targets is not None:
        loss, acc, n_correct, n_total = calc_nll_loss_and_acc(predictions,targets)
    else:
        loss,acc=None, None
    if return_str_preds:
        # Return predictions in detokenized string format
        predictions = predictions.argmax(-1)
    return predictions, loss, acc

# def vqa_struct(omninet,images,text,structured,targets=None,mode='train',return_str_preds=False,num_steps=1):
#     batch_size = images.shape[0]
#     # print('batch_size:',batch_size)
#     omninet.reset(batch_size)
#     omninet.encode_images(images,domain='IMAGE')
#     pad_mask = omninet.encode_englishtexts(text)
#     omninet.encode_structured(structured)#, pad_mask=pad_mask)
#     if mode in ['train','val']:
#         predictions = omninet.decode_from_targets('VQA_struct', targets=targets)
#     elif mode=='predict':
#         predictions = omninet.decode_greedy('VQA_struct', num_steps=num_steps)
#     # Calculate loss if targets is provided
#     if targets is not None:
#         loss, acc = calc_nll_loss_and_acc(predictions,targets)
#     else:
#         loss,acc=None, None
#     if return_str_preds:
#         # Return predictions in detokenized string format
#         predictions = predictions.argmax(-1)
#     return predictions, loss, acc


def hmdb(omninet,videos,targets=None,mode='train',return_str_preds=False,num_steps=1):
    batch_size = videos.shape[0]
    #Reset OmniNet state
    omninet.reset(batch_size)
    #Encode video files
    omninet.encode_videos(videos,domain='IMAGE')
    if mode in ['train','val']:
        predictions = omninet.decode_from_targets('HMDB',targets=targets)
    elif mode =='predict':
        predictions = omninet.decode_greedy('HMDB', num_steps=num_steps)
    # Calculate loss if targets is provided
    if targets is not None:
        loss, acc, n_correct, n_total = calc_nll_loss_and_acc(predictions,targets)
    else:
        loss,acc=None,None
    if return_str_preds:
        # Return predictions in detokenized string format
        predictions = predictions.argmax(-1)
    return predictions, loss, acc

def vqa(omninet,images,questions,structured=None,targets=None,mode='train',return_str_preds=False,num_steps=1,train_emb=False,bbox_mask=None):
    # Reset the cnp memory
    batch_size = images.shape[0]
    omninet.reset(batch_size)

    if structured is not None:
        omninet.encode_structured(structured)

    omninet.encode_images(images,domain='IMAGE',bbox_mask=bbox_mask)
    # Encode and store questions
    omninet.encode_englishtexts(questions)

    # print(omninet.cnp.spatial_cache[3,0,:10])
    # print(omninet.cnp.spatial_cache[10,-1,256:266])
    # print(omninet.cnp.spatial_cache[-1,0,-10:])

    # print(omninet.cnp.temporal_cache[0,0,:10])
    # print(omninet.cnp.temporal_cache[50,-1,256:266])
    # print(omninet.cnp.temporal_cache[-1,0,-10:])

    # if structured is not None:
    #     omninet.encode_structured(structured)

    if mode in ['train','val']:
        predictions = omninet.decode_from_targets('VQA', targets=targets)
    elif mode=='predict':
        predictions = omninet.decode_greedy('VQA', num_steps=num_steps)
    # Calculate loss if targets is provided
    if targets is not None:
        loss, acc, n_correct, n_total = calc_nll_loss_and_acc(predictions,targets)
        # loss, acc = calc_nll_loss_and_acc(predictions,targets)
        
        # if structured is not None:
        #     loss = loss + structured_spatial_emb_dist + structured_temporal_emb_dist #omninet.cos_dist()
    else:
        loss,acc=None, None
    if return_str_preds:
        # Return predictions in detokenized string format
        predictions = predictions.argmax(-1)
    return predictions, loss, acc


def vqa_fusion(omninet,images,questions,structured=None,targets=None,mode='train',return_str_preds=False,num_steps=1,train_emb=False,bbox_mask=None):
    # Reset the cnp memory
    batch_size = images.shape[0]
    omninet.reset(batch_size)

    if structured is not None:
        # print('routines')
        # print(structured.size())
        omninet.encode_structured(structured,fusion=True)

    # # Encode and store images
    # structured_spatial_emb_dist = omninet.encode_images(images,domain='IMAGE')
    # # Encode and store questions
    # structured_temporal_emb_dist = omninet.encode_englishtexts(questions)

    # Encode and store images
    structured_spatial_emb_dist, spat_struct_enc_gate = omninet.encode_images(images,domain='IMAGE',bbox_mask=bbox_mask,fusion=True)
    # Encode and store questions
    structured_temporal_emb_dist, temp_struct_enc_gate = omninet.encode_englishtexts(questions,fusion=True)

    # print(omninet.cnp.spatial_cache[3,0,:10])
    # print(omninet.cnp.spatial_cache[10,-1,256:266])
    # print(omninet.cnp.spatial_cache[-1,0,-10:])

    # print(omninet.cnp.temporal_cache[0,0,:10])
    # print(omninet.cnp.temporal_cache[50,-1,256:266])
    # print(omninet.cnp.temporal_cache[-1,0,-10:])

    # if structured is not None:
    #     omninet.encode_structured(structured)

    if mode in ['train','val']:
        predictions = omninet.decode_from_targets('VQA', targets=targets)
    elif mode=='predict':
        predictions = omninet.decode_greedy('VQA', num_steps=num_steps)
    # Calculate loss if targets is provided
    if targets is not None:
        loss, acc, n_correct, n_total = calc_nll_loss_and_acc(predictions,targets)
        # loss, acc = calc_nll_loss_and_acc(predictions,targets)
        
        # if structured is not None:
        #     loss = loss + structured_spatial_emb_dist + structured_temporal_emb_dist #omninet.cos_dist()
    else:
        loss,acc=None, None
    if return_str_preds:
        # Return predictions in detokenized string format
        predictions = predictions.argmax(-1)
    return predictions, loss, acc, structured_spatial_emb_dist, structured_temporal_emb_dist, spat_struct_enc_gate, temp_struct_enc_gate #, emb_loss

def vqa_saliency(omninet,images,questions,structured_raw=None,structured_temp=None,structured_spat=None,targets=None,mode='train',return_str_preds=False,num_steps=1,train_emb=False,bbox_mask=None,cache_grads_dict=None):
    # Reset the cnp memory
    batch_size = images.shape[0]
    omninet.reset(batch_size)

    if structured_raw is not None:
        omninet.encode_structured_saliency(structured_raw,structured_temp,structured_spat)

    omninet.encode_images(images,domain='IMAGE',bbox_mask=bbox_mask)
    # Encode and store questions
    omninet.encode_englishtexts(questions)

    if cache_grads_dict is not None:
        omninet.cnp.temporal_cache.register_hook(lambda grad: cache_grads_dict['temp'].append(grad.detach().cpu().numpy()))
        omninet.cnp.spatial_cache.register_hook(lambda grad: cache_grads_dict['spat'].append(grad.detach().cpu().numpy()))

    if mode in ['train','val']:
        predictions = omninet.decode_from_targets('VQA', targets=targets)
    elif mode=='predict':
        predictions = omninet.decode_greedy('VQA', num_steps=num_steps)
    # Calculate loss if targets is provided
    if targets is not None:
        loss, acc, n_correct, n_total = calc_nll_loss_and_acc(predictions,targets)
    else:
        loss,acc=None, None
    if return_str_preds:
        # Return predictions in detokenized string format
        predictions = predictions.argmax(-1)
    return predictions, loss, acc


def vqa_fusion_saliency(omninet,images,questions,structured_raw=None,structured_temp=None,structured_spat=None,targets=None,mode='train',return_str_preds=False,num_steps=1,train_emb=False,bbox_mask=None):
    # Reset the cnp memory
    batch_size = images.shape[0]
    omninet.reset(batch_size)

    if structured_raw is not None:
        # print('routines')
        # print(structured.size())
        # omninet.perph_encode_structured(structured)
        omninet.perph_encode_structured_saliency(structured_raw,structured_temp,structured_spat)

    # # Encode and store images
    # structured_spatial_emb_dist = omninet.encode_images(images,domain='IMAGE')
    # # Encode and store questions
    # structured_temporal_emb_dist = omninet.encode_englishtexts(questions)

    # Encode and store images
    structured_spatial_emb_dist, spat_struct_enc_gate = omninet.encode_images(images,domain='IMAGE',bbox_mask=bbox_mask,fusion=True)
    # Encode and store questions
    structured_temporal_emb_dist, temp_struct_enc_gate = omninet.encode_englishtexts(questions,fusion=True)

    # print(omninet.cnp.spatial_cache[3,0,:10])
    # print(omninet.cnp.spatial_cache[10,-1,256:266])
    # print(omninet.cnp.spatial_cache[-1,0,-10:])

    # print(omninet.cnp.temporal_cache[0,0,:10])
    # print(omninet.cnp.temporal_cache[50,-1,256:266])
    # print(omninet.cnp.temporal_cache[-1,0,-10:])

    # if structured is not None:
    #     omninet.encode_structured(structured)

    if mode in ['train','val']:
        predictions = omninet.decode_from_targets('VQA', targets=targets)
    elif mode=='predict':
        predictions = omninet.decode_greedy('VQA', num_steps=num_steps)
    # Calculate loss if targets is provided
    if targets is not None:
        loss, acc, n_correct, n_total = calc_nll_loss_and_acc(predictions,targets)
        # loss, acc = calc_nll_loss_and_acc(predictions,targets)
        
        # if structured is not None:
        #     loss = loss + structured_spatial_emb_dist + structured_temporal_emb_dist #omninet.cos_dist()
    else:
        loss,acc=None, None
    if return_str_preds:
        # Return predictions in detokenized string format
        predictions = predictions.argmax(-1)
    return predictions, loss, acc, structured_spatial_emb_dist, structured_temporal_emb_dist, spat_struct_enc_gate, temp_struct_enc_gate #, emb_loss


def image_caption(omninet,images,targets=None,mode='train',return_str_preds=False,num_steps=100):
    # Reset the cnp memory
    batch_size=images.shape[0]
    omninet.reset(batch_size)
    # Encode and store images
    omninet.encode_images(images,domain='IMAGE')
    #Calculate pad mask using the inbuilt tokenizer
    if targets is not None:
        targets,target_pad_mask = omninet.english_language_perph.tokenize_sentences(targets)
    if mode  in ['train','val']:
        predictions = omninet.decode_from_targets('IMAGE_CAPTION', targets=targets,target_pad_mask=target_pad_mask)
    elif mode=='predict':
        predictions = omninet.decode_greedy('IMAGE_CAPTION', num_steps=num_steps)
    # Calculate loss if targets is provided
    loss = None
    acc = None
    if targets is not None:
        pad_token = omninet.english_language_perph.id_PAD
        loss,acc, n_correct, n_total=calc_nll_loss_and_acc(predictions,targets,pad_id=pad_token,target_pad_mask=target_pad_mask)
    else:
        loss, acc= None, None

    if return_str_preds:
        # Return predictions in detokenized string format
        predictions = predictions.argmax(-1)
        predictions = omninet.english_language_perph.decode_tokens(predictions)
    return predictions,loss,acc

def penn(omninet,texts,pad_id=None,targets=None,target_pad_mask=None,mode='train',return_str_preds=False,num_steps=100):
    batch_size=len(texts)
    omninet.reset(batch_size)
    #Store the sentences
    omninet.encode_englishtexts(texts, domain='ENGLISH')
    #Get the tokenized targets
    if mode in ['train','val']:
        predictions = omninet.decode_from_targets('PENN', targets=targets,target_pad_mask=target_pad_mask)
    elif mode=='predict':
        predictions = omninet.decode_greedy('PENN', num_steps=num_steps)
    #Calculate loss if targets is provided
    loss=None
    acc=None
    if targets is not None:
        loss, acc, n_correct, n_total  = calc_nll_loss_and_acc(predictions,targets,pad_id=pad_id,target_pad_mask=target_pad_mask)
    else:
        loss, acc = None, None
    if return_str_preds:
    # Return predictions in detokenized string format
        predictions=predictions.argmax(-1)
    return predictions,loss,acc
 

def calc_nll_loss_and_acc(predictions, targets, pad_id=None, target_pad_mask=None, sample_weights=None):
    #Calculate loss
    pr = torch.reshape(predictions, [-1, predictions.shape[2]])
    if pad_id is not None:
        loss_fn = nn.NLLLoss(ignore_index=pad_id, reduction='none')
    else:
        loss_fn = nn.NLLLoss(reduction='none')
    targets = torch.reshape(targets, [-1])
    loss = loss_fn(pr, targets)

    # print(loss)
    # loss_fn1 = nn.NLLLoss(reduction='none')
    # loss1 = loss_fn1(pr, targets)
    # print(torch.mean(loss1))
    # print(sample_weights)
    # print(torch.mean(torch.mul(loss, sample_weights)))

    if sample_weights is not None:
        loss = torch.mean(torch.mul(loss, sample_weights))
    else:
        loss = torch.mean(loss)

    #Calculate accuracy
    preds=predictions.argmax(-1)
    preds=torch.reshape(preds,[-1])
    if target_pad_mask is not None:
        target_pad_mask=torch.reshape(target_pad_mask,[-1])
        preds=preds+(target_pad_mask*1000000).to(dtype=torch.long)
        n_total = targets.shape[0]-target_pad_mask.sum().cpu().numpy()
    else:
        n_total = targets.shape[0]
    
    n_correct = torch.sum(targets==preds).sum().cpu().numpy()
    acc=(n_correct/(n_total))*100
    
    # TODO: weighted acc
    return loss, acc, n_correct, n_total

# def calc_nll_loss_and_acc(predictions, targets, pad_id=None, target_pad_mask=None):
#     #Calculate loss
#     pr = torch.reshape(predictions, [-1, predictions.shape[2]])
#     if pad_id is not None:
#         loss_fn = nn.NLLLoss(ignore_index=pad_id)
#     else:
#         loss_fn = nn.NLLLoss()
#     targets = torch.reshape(targets, [-1])
#     loss = loss_fn(pr, targets)
#     #Calculate accuracy
#     preds=predictions.argmax(-1)
#     preds=torch.reshape(preds,[-1])
#     if target_pad_mask is not None:
#         target_pad_mask=torch.reshape(target_pad_mask,[-1])
#         preds=preds+(target_pad_mask*1000000).to(dtype=torch.long)
#         acc=(torch.sum(targets==preds).sum().cpu().numpy()/(targets.shape[0]-target_pad_mask.sum().cpu().numpy()))*100
#     else:
#         acc=(torch.sum(targets==preds).sum().cpu().numpy()/(targets.shape[0]))*100
#     return loss, acc
