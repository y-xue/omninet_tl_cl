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

OmniNet Central Neural Processor implementation

"""

from .Layers import *
from ..util import *
from torch.nn.functional import log_softmax, softmax

import numpy
import sys
numpy.set_printoptions(threshold=sys.maxsize, suppress=True)

class CNP(nn.Module):

    def __init__(self,tasks,conf=None,domains=['EMPTY'],gpu_id=-1):
        super(CNP, self).__init__()
        default_conf=self.__defaultconf__()
        if(conf!=None):
            for k in conf.keys():
                if k not in conf:
                    raise ValueError("The provided configuration does not contain %s"%k)
        else:
            conf=default_conf
        #Load the Confurigation
        self.gpu_id=gpu_id
        self.input_dim=conf['input_dim']
        self.control_dim=conf['control_dim']
        self.output_dim=conf['output_dim']
        self.spatial_dim=conf['spatial_dim']
        self.temporal_dim=conf['temporal_dim']
        self.temporal_n_layers=conf['temporal_n_layers']
        self.temporal_n_heads=conf['temporal_n_heads']
        self.temporal_d_k=conf['temporal_d_k']
        self.temporal_d_v=conf['temporal_d_v']
        self.temporal_hidden_dim=conf['temporal_hidden_dim']
        self.decoder_dim=conf['decoder_dim']
        self.decoder_n_layers=conf['decoder_n_layers']
        self.decoder_n_heads=conf['decoder_n_heads']
        self.decoder_d_k=conf['decoder_d_k']
        self.decoder_d_v=conf['decoder_d_v']
        self.decoder_hidden_dim=conf['decoder_hidden_dim']
        self.fusion_n_heads=conf['fusion_n_heads']
        self.fusion_d_k=conf['fusion_d_k']
        self.fusion_d_v=conf['fusion_d_v']
        self.fusion_hidden_dim=conf['fusion_hidden_dim']
        self.struct_dim=conf['struct_dim']
        self.logit_struct_periph_dim=conf['logit_struct_periph_dim']
        self.max_seq_len=conf['max_seq_len']
        self.output_embedding_dim=conf['output_embedding_dim']
        self.dropout=conf['dropout']
        self.batch_size=-1 #Uninitilized CNP memory
        self.use_s_decoder=conf['use_s_decoder']
        self.use_p_decoder=conf['use_p_decoder']
        self.inject_at_logits=conf['inject_at_logits']
        self.inject_at_encoder=conf['inject_at_encoder']
        self.inject_after_encoder=conf['inject_after_encoder']
        self.inject_at_decoder=conf['inject_at_decoder']
        self.temp_fusion_attn_type=conf['temp_fusion_attn_type']
        self.spat_fusion_attn_type=conf['spat_fusion_attn_type']
        self.convex_gate=conf['convex_gate']
        self.no_logit_struct_peripheral = conf['no_logit_struct_peripheral']
        # self.video_clip_len = 16
        self.img_enc_width = 7
        self.img_enc_height = 7
        # self.spatial_flat_size = self.spatial_dim * self.video_clip_len * self.img_enc_height * self.img_enc_width

        if conf['pooling']:
            print('setting spatial_flat_size to spatial_dim')
            self.spatial_flat_size = self.spatial_dim
        else:
            self.spatial_flat_size = self.spatial_dim * self.img_enc_height * self.img_enc_width

        self.temporal_flat_size = self.temporal_dim * self.max_seq_len

        # Prepare the task lists and various output classifiers and embeddings
        if isinstance(tasks, dict):
            self.task_clflen = list(tasks.values())
            self.task_dict = {t: i for i, t in enumerate(tasks.keys())}
        else:
            raise ValueError('Tasks must be of type dict containing the tasks and output classifier dimension')

        self.output_clfs = nn.ModuleList([nn.Linear(self.output_dim, t) for t in self.task_clflen])
        #Use one extra to define padding
        self.output_embs = nn.ModuleList([nn.Embedding(t+1,self.output_embedding_dim,padding_idx=t) for t in self.task_clflen])

        #Initialize the various sublayers of the CNP
        control_states=domains+list(tasks.keys())
        self.control_peripheral=ControlPeripheral(self.control_dim,control_states,gpu_id=gpu_id)
        self.temporal_encoder = TemporalCacheEncoder(self.max_seq_len,self.temporal_n_layers,
                                                     self.temporal_n_heads,self.temporal_d_k,self.temporal_d_v,
                                                    self.temporal_dim,self.temporal_hidden_dim,dropout=self.dropout,
                                                     gpu_id=self.gpu_id)
        self.decoder=Decoder(self.max_seq_len,self.decoder_n_layers,self.decoder_n_heads,self.decoder_d_k,
                             self.decoder_d_v,self.decoder_dim,self.decoder_hidden_dim,self.temporal_dim,
                             self.spatial_dim,self.output_dim, dropout=self.dropout,gpu_id=self.gpu_id)

        self.s_decoder = Decoder(self.max_seq_len,self.decoder_n_layers,self.decoder_n_heads,self.decoder_d_k,
                             self.decoder_d_v,self.decoder_dim,self.decoder_hidden_dim,self.temporal_dim,
                             self.spatial_dim,self.output_dim, dropout=self.dropout,gpu_id=self.gpu_id)

        self.p_decoder = DecoderParallel(self.max_seq_len,self.decoder_n_layers,self.decoder_n_heads,self.decoder_d_k,
                             self.decoder_d_v,self.decoder_dim,self.decoder_hidden_dim,self.temporal_dim,
                             self.spatial_dim,self.output_dim, dropout=self.dropout,gpu_id=self.gpu_id)

        # integrate structured features into unstructured encodings
        self.temporal_cache_fusion = CacheFusion(self.temporal_dim,self.fusion_n_heads,self.fusion_d_k,
            self.fusion_d_v,d_inner=self.fusion_hidden_dim,dropout=0.1,gpu_id=-1,
            attention_type=self.temp_fusion_attn_type, convex_gate=self.convex_gate)
        self.spatial_cache_fusion = CacheFusion(self.spatial_dim,self.fusion_n_heads,self.fusion_d_k,
            self.fusion_d_v,d_inner=self.fusion_hidden_dim,dropout=0.1,gpu_id=-1,
            attention_type=self.spat_fusion_attn_type, convex_gate=self.convex_gate)

        # structured feature has been projected to the same dim as temporal/spatial data
        if self.convex_gate:
            self.temporal_struct_encoder_gate = nn.Linear(self.temporal_dim + self.temporal_flat_size, 2)
            self.spatial_struct_encoder_gate = nn.Linear(self.spatial_dim + self.spatial_flat_size, 2)
        else:
            self.temporal_struct_encoder_gate = nn.Linear(self.temporal_dim + self.temporal_flat_size, 1)
            self.spatial_struct_encoder_gate = nn.Linear(self.spatial_dim + self.spatial_flat_size, 1)

        self.bn_temp = torch.nn.BatchNorm1d(self.temporal_dim + self.temporal_flat_size)
        self.bn_spat = torch.nn.BatchNorm1d(self.spatial_dim + self.spatial_flat_size)

        # # integrate unstructured encodings into structured features to produce 
        # # an overall feature vector for unstructured features and unstructured features
        # self.struct_temporal_enc = StructCacheEncoder(self.gate_n_layers, self.gate_n_head, self.gate_d_k, 
        #     self.gate_d_v, self.gate_hidden_dim, self.temporal_dim, dropout=0.1, gpu_id=-1)
        # self.struct_spatial_enc = StructCacheEncoder(self.gate_n_layers, self.gate_n_head, self.gate_d_k, 
        #     self.gate_d_v, self.gate_hidden_dim, self.spatial_dim, dropout=0.1, gpu_id=-1)

        #Initialize the various CNP caches as empty
        self.spatial_cache=None
        self.temporal_cache=None
        self.decoder_cache=None
        self.temporal_spatial_link=[]
        self.pad_cache=None    #Used to store the padding values so that it can be used later in enc dec attn
        self.structured_raw = None # (b,f)
        self.structured_temp = None
        self.structured_spat = None
        # self.struct_temp_encoding = None
        # self.struct_spat_encoding = None

        #Various projection layers
        self.spatial_pool=nn.AdaptiveAvgPool1d(1)
        self.inpcont_input_proj=nn.Linear(self.input_dim+self.control_dim,self.input_dim)
        self.input_spatial_proj=nn.Linear(self.input_dim,self.spatial_dim)
        self.input_temporal_proj=nn.Linear(self.input_dim,self.temporal_dim)
        self.emb_decoder_proj=nn.Linear(self.output_embedding_dim,self.decoder_dim)
        self.cont_decoder_proj=nn.Linear(self.control_dim,self.decoder_dim)

        if self.no_logit_struct_peripheral:
            self.structured_logit=nn.Linear(self.output_dim+self.struct_dim,self.output_dim)
        else:
            self.structured_logit=nn.Linear(self.output_dim+self.logit_struct_periph_dim,self.output_dim)

        self.structured_spatial_proj=nn.Linear(self.struct_dim,self.spatial_dim)
        self.structured_temporal_proj=nn.Linear(self.struct_dim,self.temporal_dim)
        self.struct_spat_cont_proj=nn.Linear(self.spatial_dim+self.control_dim,self.spatial_dim)
        self.struct_temp_cont_proj=nn.Linear(self.temporal_dim+self.control_dim,self.temporal_dim)
        #freeze layers

        self.cos = nn.CosineSimilarity(dim=1)

        self.structured_temporal_emb_dist = 0
        self.structured_spatial_emb_dist = 0
    
        # if self.gpu_id >= 0:
        #     self.structured_temporal_emb_dist = torch.zeros(1, device=self.gpu_id)
        #     self.structured_spatial_emb_dist = torch.zeros(1, device=self.gpu_id)
        # else:
        #     self.structured_temporal_emb_dist = torch.zeros(1)
        #     self.structured_spatial_emb_dist = torch.zeros(1)
    
        
    def decode(self,task,targets=None,num_steps=100,recurrent_steps=1,pad_mask=None,beam_width=1,return_attns=False):
        if targets is not None:
            b,t=targets.shape
            #Use teacher forcing to generate predictions. the graph is kept in memory during this operation.
            if (len(targets.shape) != 2 or targets.shape[0] != self.batch_size):
                raise ValueError(
                    "Target tensor must be of shape (batch_size,length of sequence).")
            if task not in self.task_dict.keys():
                raise ValueError('Invalid task %s'%task)
            
            dec_inputs=self.output_embs[self.task_dict[task]](targets) # (b, t, output_dim)
            dec_inputs=self.emb_decoder_proj(dec_inputs)
            control=self.control_peripheral(task,(self.batch_size)) # (b, control_dim)
            control=control.unsqueeze(1) # (b, 1, control_dim)
            control=self.cont_decoder_proj(control)
            dec_inputs=torch.cat([control,dec_inputs],1) # (b, t+1, decoder_dim)
            
            # Get output from decoder
            #Increase the length of the pad_mask to match the size after adding the control vector
            if pad_mask is not None:
                pad_extra=torch.zeros((b,1),device=self.gpu_id,dtype=pad_mask.dtype)
                pad_mask=torch.cat([pad_extra,pad_mask],1)
            
            if self.use_p_decoder:
                # print('using p_decoder')
                logits, attn_scores=self.p_decoder(dec_inputs,self.spatial_cache, self.temporal_cache,
                                 self.pad_cache, recurrent_steps=recurrent_steps,pad_mask=pad_mask,return_attns=return_attns)
            else:
                logits, attn_scores=self.decoder(dec_inputs,self.spatial_cache, self.temporal_cache,self.temporal_spatial_link,
                                 self.pad_cache, recurrent_steps=recurrent_steps,pad_mask=pad_mask,return_attns=return_attns)
            
            if self.use_s_decoder and self.inject_at_decoder:
                s_logits, s_attn_scores=self.s_decoder(self.structured_raw,self.spatial_cache, self.temporal_cache,self.temporal_spatial_link,
                                     self.pad_cache, recurrent_steps=recurrent_steps,pad_mask=pad_mask,return_attns=return_attns)
                logits = logits + s_logits

            if self.structured_raw is not None and self.inject_at_logits:
                # print('structured_raw')
                structured_raw = self.structured_raw.unsqueeze(1).expand(-1,t+1,-1)
                # print(structured_raw.size())
                # print(logits.size())
                logits = self.structured_logit(torch.cat([logits, structured_raw], 2))
                # print(logits.size())

            #Predict using the task specific classfier
            predictions=self.output_clfs[self.task_dict[task]](logits)
            predictions=predictions[:,0:t,:]

            if return_attns:
                return log_softmax(predictions,dim=2), attn_scores
            return log_softmax(predictions,dim=2)
        else:
            control = self.control_peripheral(task, (self.batch_size)) # control: (b x control_dim)
            control = control.unsqueeze(1) # (b x N x control_dim), N = 1
            control = self.cont_decoder_proj(control) # (b x N x decoder_dim), N = 1
            dec_inputs=control
            
            for i in range(num_steps-1):
                if self.use_p_decoder:
                    # print('using p_decoder')
                    logits, attn_scores=self.p_decoder(dec_inputs,self.spatial_cache, self.temporal_cache,
                                     self.pad_cache, recurrent_steps=recurrent_steps,pad_mask=pad_mask,return_attns=return_attns)
                else:
                    logits, attn_scores = self.decoder(dec_inputs, self.spatial_cache, self.temporal_cache,self.temporal_spatial_link,
                                           self.pad_cache,
                                           recurrent_steps=recurrent_steps)

                prediction = self.output_clfs[self.task_dict[task]](logits)
                prediction=prediction[:,-1,:].unsqueeze(1)
                prediction=log_softmax(prediction,dim=2).argmax(-1)
                prediction=self.output_embs[self.task_dict[task]](prediction)
                prediction = self.emb_decoder_proj(prediction).detach()
                if beam_width>1:
                    p=torch.topk(softmax(prediction),beam_width)
                    
                dec_inputs=torch.cat([dec_inputs,prediction],1) # dec_inputs: (b x (++N) x decoder_dim)

            if self.use_p_decoder:
                # print('using p_decoder')
                logits, attn_scores=self.p_decoder(dec_inputs,self.spatial_cache, self.temporal_cache,
                                 self.pad_cache, recurrent_steps=recurrent_steps, pad_mask=pad_mask, return_attns=return_attns)
            else:
                logits, attn_scores = self.decoder(dec_inputs, self.spatial_cache, self.temporal_cache,self.temporal_spatial_link,
                    self.pad_cache,recurrent_steps=recurrent_steps, return_attns=return_attns)

            if self.use_s_decoder and self.inject_at_decoder:
                # decoder_gate = self.decoder_gate(self.structured_raw)

                s_logits, s_attn_scores=self.s_decoder(self.structured_raw,self.spatial_cache, self.temporal_cache,self.temporal_spatial_link,
                                     self.pad_cache,
                                     recurrent_steps=recurrent_steps,pad_mask=pad_mask,return_attns=return_attns)
                # # TODO               
                # b,... = s_logits.shape
                # decoder_gate = decoder_gate[(..., ) + (None, ) * 2].expand(-1,len_k,n_head*dv)

                logits = logits + decoder_gate * s_logits

            if self.structured_raw is not None and self.inject_at_logits:
                structured_raw = self.structured_raw.unsqueeze(1).expand(-1,num_steps,-1)
                logits = self.structured_logit(torch.cat([logits, structured_raw], 2))
                
            predictions = self.output_clfs[self.task_dict[task]](logits)

            if return_attns:
                return log_softmax(predictions,dim=2), attn_scores
            return log_softmax(predictions,dim=2)

    def encode(self,input,pad_mask=None,domain='EMPTY',recurrent_steps=1,features=None,bbox_mask=None):
        if (len(input.shape)!=4):
            raise Exception('Invalid input dimensions.')
        b,t,s,f=list(input.size()) # input: b x t x s x f (f: feature size (m_model))
        if domain == 'STRUCT_SPAT':
            self.temporal_spatial_link.append((0,1))
        elif domain == 'STRUCT_TEMP':
            self.temporal_spatial_link.append((1,0))
        else:
            self.temporal_spatial_link.append((t,s))
        if b != self.batch_size:
            raise Exception('Input batch size does not match.')
        #Spatial encode. Spatial encodes encodes both spatial and time dimension features together
        if 'STRUCT' in domain:
            input = input.squeeze(1)
            # print('0 input:', input.shape)
            control_vecs = self.control_peripheral(domain, (b)) # b x control_dim
            control_vecs=control_vecs.unsqueeze(1) # b x 1 x control_dim
            # print('1 control_vecs:', control_vecs.shape)
            input = torch.cat([input, control_vecs], 2)
            # print('2 input:', input.shape)
        else:
            control_vecs = self.control_peripheral(domain, (b, t, s)) # control_vecs: b x t x s x control_dim
            input = torch.cat([input, control_vecs], 3) # input: b x t x s x (control_dim+f)
        
        input=self.inpcont_input_proj(input) # projected to: b x t x s x f (or b x 1 x f for structured)

        # print('domain:', domain)
        #Project the spatial data, into the query dimension and add it to the existing cache
        if s>1 or domain == 'STRUCT_SPAT':
            if s>1:
                spatial_f=torch.reshape(input,[b,t*s,f])
                spatial_f=self.input_spatial_proj(spatial_f)
            else:
                spatial_f=self.input_spatial_proj(input)
            
            # print('0 spatial_f:', spatial_f.shape)
            if self.spatial_cache is None:
                self.spatial_cache=spatial_f
            else:
                self.spatial_cache=torch.cat([self.spatial_cache,spatial_f],1)
            # print('1 spatial_cache:', self.spatial_cache.shape)

        if domain == 'STRUCT_SPAT':
            return

        if domain == 'STRUCT_TEMP':
            temp_data=self.input_temporal_proj(input)
        else:
            #Feed the time features. First AVGPool the spatial features.
            temp_data=input.transpose(2,3).reshape(b*t,f,s)
            temp_data=self.spatial_pool(temp_data).reshape(b,t,f)
            temp_data=self.input_temporal_proj(temp_data)
            #Create a control state and concat with the temporal data
            #Add data to temporal cache
            temp_data,=self.temporal_encoder(temp_data,pad_mask=pad_mask,recurrent_steps=recurrent_steps)

        # print('1 temp_data:', temp_data.shape)
        if self.temporal_cache is None:
            self.temporal_cache=temp_data
        else:
            self.temporal_cache=torch.cat([self.temporal_cache,temp_data],1)
        # print('2 temporal_cache:', self.temporal_cache.shape)

        #Add pad data to pad cache
        if pad_mask is None:
            pad_mask=torch.zeros((b,t),device=self.gpu_id,dtype=torch.uint8)
        if self.pad_cache is None:
            self.pad_cache=pad_mask
        else:
            self.pad_cache=torch.cat([self.pad_cache,pad_mask],1)
            

    def encode_fusion(self,input,pad_mask=None,domain='EMPTY',recurrent_steps=1,features=None,bbox_mask=None):
        if (len(input.shape)!=4):
            raise Exception('Invalid input dimensions.')
        b,t,s,f=list(input.size()) # input: b x t x s x f (f: feature size (m_model))
        self.temporal_spatial_link.append((t,s))
        if b != self.batch_size:
            raise Exception('Input batch size does not match.')

        emb_dist = None
        struct_enc_gate = None
        # calculate cos distance after perph and before actual encoding 
        if (self.inject_at_encoder or self.inject_after_encoder) and self.structured_temp is not None and self.structured_spat is not None:
            if domain=='ENGLISH':
                temporal_f=input.transpose(2,3).reshape(b*t,f,s) # temporal_f = input.squeeze(2) # (b, t, f)
                temporal_f=self.spatial_pool(temporal_f).reshape(b,t,f)
                # change to +=
                emb_dist = self.cos_dist(temporal_f, self.structured_temp, pad_mask=pad_mask)
            
            if domain=='IMAGE':
                spatial_f = torch.reshape(input,[b,t*s,f]) # (b, t*s, f)
                # change to +=
                emb_dist = self.cos_dist(spatial_f, self.structured_spat)


        #Spatial encode. Spatial encodes encodes both spatial and time dimension features together
        control_vecs = self.control_peripheral(domain, (b, t, s)) # control_vecs: b x t x s x control_dim
        # print('control_vecs:')
        # print(control_vecs[0,0,0,:10])
        # print(control_vecs[10,-1,-1,-10:])
        # print(control_vecs[-1,0,-1,:10])
        input = torch.cat([input, control_vecs], 3) # input: b x t x s x (control_dim+f)
        input=self.inpcont_input_proj(input) # projected to: b x t x s x f

        #Project the spatial data, into the query dimension and add it to the existing cache
        if s>1:
            spatial_f=torch.reshape(input,[b,t*s,f])
            spatial_f=self.input_spatial_proj(spatial_f)

            # print('0 spatial_f:', spatial_f.shape)

            # fuse images or videos with structured  
            if (self.inject_at_encoder or self.inject_after_encoder) and self.structured_spat is not None and domain == 'IMAGE':
                structured_spat = self.encode_structured(self.structured_spat, domain=domain)
                structured_spat = self.struct_spat_cont_proj(structured_spat)

                # print('1 structured_spat:', structured_spat.shape)

                if self.spat_fusion_attn_type == 'selective':
                    spatial_f_lst = []

                    for ti in range(0,t*s,s):
                        spatial_f_ti = spatial_f[:,ti:ti+s,:]

                        # self.struct_temp_encoding = self.struct_temporal_enc()
                        if features is None:
                            spatial_flat = torch.reshape(spatial_f_ti, [b,-1])
                        else:
                            # print('features:', features.shape)
                            spatial_flat = features
                        # print('spatial_flat:', spatial_flat.shape)

                        # spatial_flat_paddings = torch.zeros((b,self.spatial_flat_size-spatial_flat.shape[1]),device=self.gpu_id,dtype=torch.float32)
                        # # print('spatial_flat_paddings:', spatial_flat_paddings.shape)

                        # print('structured_spat:', self.structured_spat.shape)
                        structured_spat_flat = torch.reshape(structured_spat, [b,-1])
                        # print('structured_spat_flat:', structured_spat_flat.shape)

                        # print('structured_spat_flat mean:', torch.mean(structured_spat_flat, dim=0).detach().cpu().numpy())
                        # print('spatial_flat mean:', torch.mean(spatial_flat, dim=0).detach().cpu().numpy())

                        gate_input = torch.cat([structured_spat_flat, spatial_flat],1)

                        # print('2 gate_input[0]:', gate_input[0])
                        # print('2 gate_input:', gate_input.shape)
                        # print('2 gate_input:', gate_input.detach().cpu().numpy())

                        gate_input_norm = self.bn_spat(gate_input)
                        # print('2 gate_input_norm[0]:', gate_input_norm[0])
                        
                        # print('layer weights:', self.spatial_struct_encoder_gate.weight.detach().cpu().numpy())
                        # print('layer bias:', self.spatial_struct_encoder_gate.bias.detach().cpu().numpy())
                        raw_gate = self.spatial_struct_encoder_gate(gate_input_norm)
                        # print('raw_gate:', raw_gate.detach().cpu().numpy())

                        # norm_gate = self.bn(raw_gate)
                        # print('norm_gate:', norm_gate.detach().cpu().numpy())

                        # print('3 raw_gate:', raw_gate)
                        if self.convex_gate:
                            struct_enc_gate = softmax(raw_gate, dim=-1)
                        else:
                            struct_enc_gate = torch.sigmoid(raw_gate)
                        # print('4 struct_enc_gate:', struct_enc_gate)
                        # struct_enc_gate = torch.sigmoid(self.spatial_struct_encoder_gate(torch.cat([structured_spat_flat, spatial_flat, spatial_flat_paddings],1)))
                        # print('spat_struct_enc_gate', spat_struct_enc_gate)
                        
                        spatial_f_ti,_ = self.spatial_cache_fusion(structured_spat, spatial_f_ti, struct_enc_gate, pad_mask=pad_mask, bbox_mask=bbox_mask)
                        spatial_f_lst.append(spatial_f_ti)

                    spatial_f = torch.cat(spatial_f_lst, 1)

                elif self.spat_fusion_attn_type == 'gated':
                    spatial_f, fusion_attn_scores = self.spatial_cache_fusion(structured_spat, spatial_f, None, return_attns=True, pad_mask=pad_mask, bbox_mask=bbox_mask)
                    # squeeze the key dim (1 for structured data)
                    # average over head dim
                    # head_mean_fusion_attn_scores = torch.mean(fusion_attn_scores.squeeze(), dim=1)
                    head_mean_fusion_attn_scores = fusion_attn_scores.squeeze()

                    mean_fusion_attn_scores = torch.mean(head_mean_fusion_attn_scores,dim=-1)
                    # mean_fusion_attn_scores = fusion_attn_scores

                elif self.spat_fusion_attn_type == 'none':
                    spatial_f,_ = self.spatial_cache_fusion(structured_spat, spatial_f, None, return_attns=False, pad_mask=pad_mask)
                else:
                    raise("Wrong spat_fusion_attn_type")

            if self.spatial_cache is None:
                self.spatial_cache=spatial_f
            else:
                self.spatial_cache=torch.cat([self.spatial_cache,spatial_f],1)

        #Feed the time features. First AVGPool the spatial features.
        temp_data=input.transpose(2,3).reshape(b*t,f,s)
        temp_data=self.spatial_pool(temp_data).reshape(b,t,f)
        temp_data=self.input_temporal_proj(temp_data)

        if (self.inject_at_encoder or self.inject_after_encoder) and self.structured_temp is not None and (domain in ['ENGLISH', 'GERMAN']):
            structured_temp = self.encode_structured(self.structured_temp, domain=domain)
            structured_temp = self.struct_temp_cont_proj(structured_temp)

            if self.inject_after_encoder:
                #Create a control state and concat with the temporal data
                #Add data to temporal cache
                temp_data,=self.temporal_encoder(temp_data,pad_mask=pad_mask,recurrent_steps=recurrent_steps)

            if self.temp_fusion_attn_type == 'selective':
                # fuse text with structured 
                if features is None:
                    temporal_flat = torch.reshape(temp_data, [b,-1])
                else:
                    # print('features:', features.shape)
                    temporal_flat = features
                # print('temporal_flat:', temporal_flat.shape)

                temporal_flat_paddings = torch.zeros((b,self.temporal_flat_size-temporal_flat.shape[1]),device=self.gpu_id,dtype=torch.float32)
                # print('temporal_flat_paddings:', temporal_flat_paddings.shape)

                # print('structured_temp:', self.structured_temp.shape)
                structured_temp_flat = torch.reshape(structured_temp, [b,-1])
                # print('structured_temp_flat:', structured_temp_flat.shape)

                gate_input = torch.cat([structured_temp_flat, temporal_flat, temporal_flat_paddings],1)
                # print('t0 gate_input:', gate_input.shape)

                gate_input_norm = self.bn_temp(gate_input)

                raw_gate =  self.temporal_struct_encoder_gate(gate_input_norm)
                # print('t1 raw_gate:', raw_gate)

                if self.convex_gate:
                    struct_enc_gate = softmax(raw_gate, dim=-1)
                else:
                    struct_enc_gate = torch.sigmoid(raw_gate)
                # print('t2 struct_enc_gate:', struct_enc_gate)
                # print('temp_struct_enc_gate', temp_struct_enc_gate)

                temp_data,_ = self.temporal_cache_fusion(structured_temp, temp_data, struct_enc_gate, pad_mask=pad_mask)
            elif self.temp_fusion_attn_type == 'gated':
                temp_data, fusion_attn_scores = self.temporal_cache_fusion(structured_temp, temp_data, None, return_attns=True, pad_mask=pad_mask)

                # squeeze the key dim (1 for structured data)
                # average over head dim
                head_mean_fusion_attn_scores = torch.mean(fusion_attn_scores.squeeze(), dim=1)
                print('temp head_mean_fusion_attn_scores:', head_mean_fusion_attn_scores[-1])
                print('temp pad_mask:', pad_mask[-1])
                non_pad_mask = pad_mask.ne(1).float()
                head_mean_fusion_attn_scores *= non_pad_mask
                mean_fusion_attn_scores = head_mean_fusion_attn_scores.sum(-1) / non_pad_mask.sum(-1)
                print('temp mean_fusion_attn_scores:', mean_fusion_attn_scores[-1])
            elif self.temp_fusion_attn_type == 'none':
                temp_data, _ = self.temporal_cache_fusion(structured_temp, temp_data, None, return_attns=False, pad_mask=pad_mask)
            else:
                raise("Wrong temp_fusion_attn_type")

            if self.inject_at_encoder:
                #Create a control state and concat with the temporal data
                #Add data to temporal cache
                temp_data,=self.temporal_encoder(temp_data,pad_mask=pad_mask,recurrent_steps=recurrent_steps)
        else:
            temp_data,=self.temporal_encoder(temp_data,pad_mask=pad_mask,recurrent_steps=recurrent_steps)

        if self.temporal_cache is None:
            self.temporal_cache=temp_data
        else:
            self.temporal_cache=torch.cat([self.temporal_cache,temp_data],1)

        #Add pad data to pad cache
        if pad_mask is None:
            pad_mask=torch.zeros((b,t),device=self.gpu_id,dtype=torch.uint8)
        if self.pad_cache is None:
            self.pad_cache=pad_mask
        else:
            self.pad_cache=torch.cat([self.pad_cache,pad_mask],1)

        # print(pad_mask.shape)
        # print('pad_mask:', pad_mask)
        # print('return struct_enc_gate', struct_enc_gate)
        if (self.temp_fusion_attn_type == 'gated' and (domain in ['ENGLISH', 'GERMAN'])) or (self.spat_fusion_attn_type == 'gated' and domain == 'IMAGE'):
            return emb_dist, mean_fusion_attn_scores

        elif (self.temp_fusion_attn_type == 'none' and domain in ['ENGLISH', 'GERMAN']) or (self.spat_fusion_attn_type == 'none' and domain == 'IMAGE'):
            return emb_dist, None

        if self.convex_gate:
            return emb_dist, struct_enc_gate[:,0]

        return emb_dist, struct_enc_gate # structured_temporal_emb_dist, structured_spatial_emb_dist
            
    def cos_dist(self, unstructured, structured, pad_mask=None):
        b,l,_ = unstructured.size()
        
        if self.gpu_id >= 0:
            dist = torch.zeros(b, device=self.gpu_id)
        else:
            dist = torch.zeros(b)

        non_pad_mask=get_non_pad_mask(unstructured,pad_mask)
        if non_pad_mask is not None:
            non_pad_mask = non_pad_mask.squeeze(-1)

        for j in range(l):
            cdist = 1 - self.cos(unstructured[:,j,:], structured[:,0,:])
            
            if non_pad_mask is not None:
                cdist *= non_pad_mask[:,j]

            dist += cdist

        if non_pad_mask is not None:
            return (dist/non_pad_mask.sum(dim=1)).mean() # change to count of non_pad

        return (dist/l).mean()
        
    # def structured_emb(self, structured):
    #     self.structured_raw = structured
    #     structured = structured.unsqueeze(1)
    #     self.structured_spat = self.structured_spatial_proj(structured)
    #     self.structured_temp = self.structured_temporal_proj(structured)

    def structured_emb_saliency(self, structured_raw, structured_temp, structured_spat):
        self.structured_raw = structured_raw

        structured_temp = structured_temp.unsqueeze(1)
        structured_spat = structured_spat.unsqueeze(1)
        self.structured_spat = self.structured_spatial_proj(structured_spat)
        self.structured_temp = self.structured_temporal_proj(structured_temp)

    def encode_structured(self,structured,domain='STRUCT'):
        b = structured.size()[0]
        control = self.control_peripheral(domain, (b)) # b x control_dim
        control=control.unsqueeze(1) # b x 1 x control_dim
        structured = torch.cat([structured, control], 2)
        return structured

    # def fusion(self, unstructured, structured):
    #     self.temporal_cache_fusion(structured, self.temporal_cache, return_attns=True, pad_mask=pad_mask)

    # def fusion(self,structured,pad_mask=None,domain='STRUCT'):
    #     structured = structured.unsqueeze(1) # # b x 1 x d_struct
    #     structured = self.structured_proj(structured)

    #     b,t,ds=list(structured.size())
    #     control = self.control_peripheral(domain, (b)) # b x control_dim
    #     control=control.unsqueeze(1) # b x 1 x control_dim
    #     structured = torch.cat([structured, control], 2)
    #     structured=self.structcont_struct_proj(structured)
    #     # if self.struct_dim != self.decoder_dim:
    #     #     structured = self.structured_proj(structured)
    #     # else:
    #     #     print('skipping structured_proj')
    #     #     print(structured.shape)
    #     self.structured_vec = structured
    #     self.temporal_cache, _ = self.temporal_cache_fusion(structured, self.temporal_cache, return_attns=True, pad_mask=pad_mask)
    #     self.spatial_cache, _ = self.spatial_cache_fusion(structured, self.spatial_cache, return_attns=True, pad_mask=pad_mask)

    # def fusion_1(self,structured,pad_mask=None,domain='STRUCT'):
    #     self.temporal_cache, _ = self.temporal_cache_fusion(structured[:,0].unsqueeze(1), self.temporal_cache, return_attns=True, pad_mask=pad_mask)
    #     self.spatial_cache, _ = self.spatial_cache_fusion(structured[:,1].unsqueeze(1), self.spatial_cache, return_attns=True, pad_mask=pad_mask)

    def clear_spatial_cache(self):
        self.spatial_cache=None

    def clear_temporal_cache(self):
        self.temporal_raw_cache=None
        self.temporal_cache=None

    def reset(self,batch_size=1):
        self.attn_scores=[]
        self.batch_size=batch_size
        self.temporal_spatial_link=[]
        self.pad_cache=None
        self.clear_spatial_cache()
        self.clear_temporal_cache()
        self.structured_raw = None
        self.structured_temp = None
        self.structured_spat = None
    
    @staticmethod
    def __defaultconf__():
        conf={
            'input_dim':128,
            'control_dim':32,
            'output_dim':128,
            'spatial_dim':128,
            'temporal_dim':512,
            'temporal_n_layers':6,
            'temporal_n_heads':8,
            'temporal_d_k':64,
            'temporal_d_v':64,
            'temporal_hidden_dim':2048,
            'decoder_dim':512,
            'decoder_n_layers':6,
            'decoder_n_heads':8,
            'decoder_d_k':64,
            'decoder_d_v':64,
            'decoder_hidden_dim':2048,
            'max_seq_len':1000,
            'output_embedding_dim':300,
            'dropout':0.1
        }
        return conf

class ControlPeripheral(nn.Module):
    """
        A special peripheral used to help the CNP identify the data domain or specify the context of
        the current operation.

    """

    # E.g.
    # domains: ['ENGLISH','GERMAN','IMAGE']
    # tasks: {'PENN':48,'HMDB':52,'IMAGE_CAPTION':25000,'VQA':3500} # task_name: output_dim
    # control_state: ['ENGLISH', 'GERMAN', 'IMAGE', 'PENN', 'HMDB', 'IMAGE_CAPTION', 'VQA']
    # control_dict: {'ENGLISH': 0, 'GERMAN': 1, 'IMAGE': 2, 'PENN': 3, 'HMDB': 4, 'IMAGE_CAPTION': 5, 'VQA': 6}
    def __init__(self, control_dim, control_states, gpu_id=-1):
        """
            Accepts as input control states as list of string. The control states are sorted before id's
            are assigned
        """
        super(ControlPeripheral, self).__init__()
        self.control_dim = control_dim
        self.gpu_id = gpu_id
        self.control_dict = {}
        for i, c in enumerate(control_states):
            self.control_dict[c] = i
        self.control_embeddings=nn.Embedding(len(control_states)+1,self.control_dim)

    # e.g.
    # control_state = domain: 'English'
    # shape (b,t,s): n_sentences * max_len * 1
    # control_emb: n_sentences * max_len * 1 * control_dim
    def forward(self, control_state, shape=()):
        if self.gpu_id>=0:
            control_ids = torch.ones(shape, dtype=torch.long,device=self.gpu_id)*self.control_dict[control_state]
        else:
            control_ids = torch.ones(shape, dtype=torch.long)*self.control_dict[control_state]
        return self.control_embeddings(control_ids)

class CacheFusion(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, d_inner=None, dropout=0.1, gpu_id=-1, attention_type='default', convex_gate=False):
        super().__init__()
        self.layer = FusionLayer(d_model, n_head, d_k, d_v, d_inner=d_inner, dropout=dropout, attention_type=attention_type, convex_gate=convex_gate)
        self.gpu_id = gpu_id

    def forward(self, structured, cache, struct_gate=None, return_attns=False, pad_mask=None, bbox_mask=None):
        # print('pad_mask:', pad_mask)
        # print('bbox_mask:', bbox_mask)
        if pad_mask is not None and bbox_mask is not None:
            mask = pad_mask | bbox_mask
            # print('both None')
        elif pad_mask is not None:
            # print('bbox_mask None')
            mask = pad_mask
        else:
            # print('pad_mask None')
            mask = bbox_mask

        if mask is not None:
            attn_mask=get_attn_key_pad_mask(mask,structured)
        else:
            attn_mask=None
        non_pad_mask=get_non_pad_mask(cache,pad_mask)

        fusion_output, fusion_attn_score = self.layer(structured, cache, non_pad_mask, struct_gate, fusion_attn_mask=attn_mask)
        if return_attns:
            return fusion_output, fusion_attn_score
        return fusion_output, -1

class TemporalCacheEncoder(nn.Module):
    def __init__(
            self,
            len_max_seq,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1,gpu_id=-1):

        super().__init__()

        n_position = len_max_seq + 1
        self.dropout_emb = nn.Dropout(dropout)
        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_model, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.gpu_id=gpu_id

    def forward(self, src_seq, return_attns=False,recurrent_steps=1, pad_mask=None):

        enc_slf_attn_list = []
        b,t,_=src_seq.shape

        if self.gpu_id >= 0:
            src_pos = torch.arange(1, t + 1,device=self.gpu_id).repeat(b, 1)
            # struct_gate = torch.ones(b, dtype=torch.long,device=self.gpu_id)
        else:
            src_pos = torch.arange(1, t + 1).repeat(b, 1)
            # struct_gate = torch.ones(b, dtype=torch.long)

        # -- Forward
        enc_output = src_seq + self.position_enc(src_pos)
        enc_output = self.dropout_emb(enc_output)
        if pad_mask is not None:
            slf_attn_mask=get_attn_key_pad_mask(pad_mask,src_seq)
        else:
            slf_attn_mask=None
        non_pad_mask=get_non_pad_mask(src_seq,pad_mask)
        for i in range(recurrent_steps):
            for enc_layer in self.layer_stack:
                enc_output, enc_slf_attn = enc_layer(
                    enc_output,non_pad_mask,slf_attn_mask=slf_attn_mask)

                if return_attns:
                    enc_slf_attn_list += [enc_slf_attn]
        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):

    def __init__(
            self,
            len_max_seq,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, temporal_dim, spatial_dim,output_dim,dropout=0.1,gpu_id=-1):

        super().__init__()
        n_position = len_max_seq + 1

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_model, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, temporal_dim,
                         spatial_dim,dropout=dropout,gpu_id=gpu_id)
            for _ in range(n_layers)])
        self.output_fc=nn.Linear(d_model,output_dim)
        self.gpu_id=gpu_id

    def forward(self, dec_inputs, spatial_cache, temporal_cache,temporal_spatial_link,
                pad_cache,
                pad_mask=None,return_attns=False,recurrent_steps=1):

        # -- Forward
        b,t,_=dec_inputs.shape
        # print('Decoder')
        # print(dec_inputs.size())
        if self.gpu_id >= 0:
            dec_pos = torch.arange(1, t + 1,device=self.gpu_id).repeat(b, 1)
        else:
            dec_pos = torch.arange(1, t + 1).repeat(b, 1)

        # if struct_gate is None:
        #     if self.gpu_id >= 0:
        #         struct_gate = torch.ones(b, dtype=torch.long,device=self.gpu_id)
        #     else:
        #         struct_gate = torch.ones(b, dtype=torch.long)

        dec_outputs = dec_inputs + self.position_enc(dec_pos) # output of position_enc: b x t x d_model
        slf_attn_mask_subseq=get_subsequent_mask((b,t),self.gpu_id)
        if pad_mask is not None:
            slf_attn_mask_keypad=get_attn_key_pad_mask(pad_mask,dec_inputs)
            slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        else:
            slf_attn_mask=slf_attn_mask_subseq
        #Run all the layers of the decoder building the prediction graph
        dec_enc_attn_mask=get_attn_key_pad_mask(pad_cache, dec_inputs)
        non_pad_mask=get_non_pad_mask(dec_inputs,pad_mask)
        for i in range(recurrent_steps):
            for dec_layer in self.layer_stack:
                dec_outputs, attns = dec_layer(dec_outputs,temporal_cache, spatial_cache,temporal_spatial_link,
                                               non_pad_mask,slf_attn_mask=slf_attn_mask,dec_enc_attn_mask=dec_enc_attn_mask)
                # print(dec_outputs.size())
        dec_outputs=self.output_fc(dec_outputs)
        if return_attns:
            return dec_outputs,attns
        return dec_outputs,[]


class DecoderParallel(nn.Module):

    def __init__(
            self,
            len_max_seq,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, temporal_dim, spatial_dim,output_dim,dropout=0.1,gpu_id=-1):

        super().__init__()
        n_position = len_max_seq + 1

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_model, padding_idx=0),
            freeze=True)

        self.temporal_layer_stack = nn.ModuleList([
            DecoderLayerSingleCache(d_model, d_inner, n_head, d_k, d_v, temporal_dim,
                         dropout=dropout,gpu_id=gpu_id)
            for _ in range(n_layers)])

        self.spatial_layer_stack = nn.ModuleList([
            DecoderLayerSingleCache(d_model, d_inner, n_head, d_k, d_v, spatial_dim,
                         dropout=dropout,gpu_id=gpu_id)
            for _ in range(n_layers)])

        # self.output_fc=nn.Linear(d_model,output_dim)
        self.parallel_output_fc=nn.Linear(2*d_model,output_dim)
        self.gpu_id=gpu_id

    def forward(self, dec_inputs, spatial_cache, temporal_cache,
                pad_cache,
                pad_mask=None,return_attns=False,recurrent_steps=1):

        # -- Forward
        b,t,_=dec_inputs.shape
        if self.gpu_id >= 0:
            dec_pos = torch.arange(1, t + 1,device=self.gpu_id).repeat(b, 1)
        else:
            dec_pos = torch.arange(1, t + 1).repeat(b, 1)
        
        slf_attn_mask_subseq=get_subsequent_mask((b,t),self.gpu_id)
        if pad_mask is not None:
            slf_attn_mask_keypad=get_attn_key_pad_mask(pad_mask,dec_inputs)
            slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        else:
            slf_attn_mask=slf_attn_mask_subseq
        #Run all the layers of the decoder building the prediction graph
        dec_enc_attn_mask=get_attn_key_pad_mask(pad_cache, dec_inputs)
        non_pad_mask=get_non_pad_mask(dec_inputs,pad_mask)

        temp_dec_outputs = dec_inputs + self.position_enc(dec_pos) # output of position_enc: b x t x d_model
        for i in range(recurrent_steps):
            for dec_layer in self.temporal_layer_stack:
                temp_dec_outputs, temp_attns = dec_layer(temp_dec_outputs,temporal_cache,
                                               non_pad_mask,slf_attn_mask=slf_attn_mask,dec_enc_attn_mask=dec_enc_attn_mask)
        
        spat_dec_outputs = dec_inputs + self.position_enc(dec_pos) # output of position_enc: b x t x d_model
        if spatial_cache is not None:
            # print('temp_dec_outputs:, ', temp_dec_outputs.shape)
            # print('temporal_cache:, ', temporal_cache.shape)
            # print('spatial_cache:, ', spatial_cache.shape)
            # print('dec_enc_attn_mask: ', dec_enc_attn_mask.shape)
            for i in range(recurrent_steps):
                for dec_layer in self.spatial_layer_stack:
                    spat_dec_outputs, spat_attns = dec_layer(spat_dec_outputs,spatial_cache,
                                                   non_pad_mask,slf_attn_mask=slf_attn_mask)
            
        dec_outputs = torch.cat([temp_dec_outputs, spat_dec_outputs], 2)
        dec_outputs = self.parallel_output_fc(dec_outputs)
        # else:
        #     dec_outputs = temp_dec_outputs

        # dec_outputs=self.output_fc(dec_outputs)
        
        if not return_attns:
            return dec_outputs,[]
        
        if spatial_cache is not None:
            return dec_outputs, temp_attns, spat_attns
        
        return dec_outputs, temp_attns
        

# class StructCacheEncoder(nn.Module):
#     def __init__(
#             self,
#             n_layers, n_head, d_k, d_v,
#             d_inner, cache_dim,dropout=0.1,gpu_id=-1):

#         super().__init__()

#         self.layer_stack = nn.ModuleList([
#             GateLayer(d_inner, n_head, d_k, d_v, cache_dim,
#                          dropout=dropout,gpu_id=gpu_id)
#             for _ in range(n_layers)])
        
#         self.gpu_id=gpu_id

#     def forward(self, structured, cache, pad_cache,
#                 return_attns=False,recurrent_steps=1):
#         # -- Forward
#         struct_enc_attn_mask=get_attn_key_pad_mask(pad_cache, structured)
        
#         for i in range(recurrent_steps):
#             for dec_layer in self.layer_stack:
#                 structured, attns = dec_layer(structured, cache,
#                                               struct_enc_attn_mask=struct_enc_attn_mask)
#         if return_attns:
#             return structured,attns
#         return structured,[]

