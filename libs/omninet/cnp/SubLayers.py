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

OmniNet transformer sub layers

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelectiveScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2) # softmax along k's 

    def forward(self, q, k, v, mask=None,k_gate=None):
        # nb, lk, dk = k.shape
        # attn = torch.bmm(k, q.transpose(1, 2)) # [(n*b) x lk x dk] dot_product [(n*b) x dk x lq] = [(n*b) x lk x lq]
        attn = torch.bmm(q, k.transpose(1, 2)) # [(n*b) x lq x dk] dot_product [(n*b) x dk x lk] = [(n*b) x lq x lk]
        # print('attention')
        # print(attn.size())

        attn = attn / self.temperature
        if k_gate is not None:
            attn=torch.mul(attn,k_gate)
        if mask is not None:
            # print(mask.size())
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn) # row-wise softmax
        # print('attn:', attn)
        # print('attn[0]:', attn[0])
        attn = self.dropout(attn) 

        # attn_expd = attn.expand(nb, lk, dk)
        # v_expd = v.expand(nb, lk, dk)
        # output = attn_expd * v_expd
        # output = torch.bmm(attn, v) # [(n*b) x lk * lq] dot_product [(n*b) x lv x dv] (where lq == lv) = [(n*b) x lk x dv]
        output = torch.bmm(attn.transpose(1,2), v) # [(n*b) x lq * lk]^T dot_product [(n*b) x lv x dv] (where lq == lv) = [(n*b) x lk x dv]

        return output, attn

class SelectiveScaledDotProductGatedAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()

        # print('SelectiveScaledDotProductGatedAttention')
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, q, k, v, mask=None,k_gate=None):
        # nb, lk, dk = k.shape
        # attn = torch.bmm(k, q.transpose(1, 2)) # [(n*b) x lk x dk] dot_product [(n*b) x dk x lq] = [(n*b) x lk x lq]
        attn = torch.bmm(q, k.transpose(1, 2)) # [(n*b) x lq x dk] dot_product [(n*b) x dk x lk] = [(n*b) x lq x lk]
        # print('attention')
        # print(attn.size())

        attn = attn / self.temperature
        if k_gate is not None:
            attn=torch.mul(attn,k_gate)
        if mask is not None:
            # print(mask.size())
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.dropout(attn)
        attn = self.sigmoid(attn)

        if torch.any(attn>1):
            print('attn:', attn)

        # attn_expd = attn.expand(nb, lk, dk)
        # v_expd = v.expand(nb, lk, dk)
        # output = attn_expd * v_expd
        # output = torch.bmm(attn, v) # [(n*b) x lk * lq] dot_product [(n*b) x lv x dv] (where lq == lv) = [(n*b) x lk x dv]
        output = torch.bmm(attn.transpose(1,2), v) # [(n*b) x lq * lk]^T dot_product [(n*b) x lv x dv] (where lq == lv) = [(n*b) x lk x dv]

        return output, attn

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None,k_gate=None):
        attn = torch.bmm(q, k.transpose(1, 2)) # [(n*b) x lq x dk] dot_product [(n*b) x dk x lk] = [(n*b) x lq x lk]
        attn = attn / self.temperature

        # print('attn: ', attn.shape)

        if k_gate is not None:
            attn=torch.mul(attn,k_gate)
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn) # row-wise softmax
        attn = self.dropout(attn) 
        output = torch.bmm(attn, v) # [(n*b) x lq * lk] dot_product [(n*b) x lv x dv] (where lk == lv) = [(n*b) x lq x dv]

        return output, attn

class MultiHeadFusionAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, attention_type='selective', convex_gate=False):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention_type = attention_type
        self.convex_gate = convex_gate
        if attention_type == 'selective':
            self.attention = SelectiveScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        elif attention_type == 'gated':
            self.attention = SelectiveScaledDotProductGatedAttention(temperature=np.power(d_k, 0.5), attn_dropout=0.5)
        else:
            self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, struct_gate=None, mask=None,k_gate=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        if k_gate is not None:
            k_gate = k_gate.transpose(0, 1)
            k_gate=k_gate.reshape(n_head*sz_b,len_q,len_v)

        residual = k

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        # v = v.view(sz_b, len_v, n_head, d_v)
        
        #A Weighting score for the keys is provided
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        if self.attention_type == 'none':
            output = v.repeat(1,len_k,1)
            attn = None
        else:
            if mask is not None:
                # print('MHA mask.shape: ', mask.size())
                mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
                # print('MHA mask.shape after repeat: ', mask.size())
            output, attn = self.attention(q, k, v, mask=mask,k_gate=k_gate)
            attn=attn.view(n_head,sz_b,len_q,len_k).transpose(0,1)

        output = output.view(n_head, sz_b, len_k, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_k, -1) # b x lq x (n*dv)
    
        output = self.dropout(self.fc(output))

        if struct_gate is not None:
            if self.convex_gate:
                g0 = struct_gate[:,0].view(-1,1).unsqueeze(2).expand(-1,len_k,n_head*d_v)
                g1 = struct_gate[:,1].view(-1,1).unsqueeze(2).expand(-1,len_k,n_head*d_v)
                residual = residual * g1
            else:
                g0 = struct_gate.unsqueeze(2).expand(-1,len_k,n_head*d_v)

            # #struct_gate = struct_gate[(..., ) + (None, ) * 2].expand(-1,len_k,n_head*d_v)
            # struct_gate = struct_gate.unsqueeze(2).expand(-1,len_k,n_head*d_v)
            # # print('MHA struct_gate:', struct_gate.shape)
            # # print('MHA output:', output.shape)
            # output = output * struct_gate
            output = output * g0

            # if self.convex_gate:
            #     residual = residual * g1
        
        output = self.layer_norm(output + residual)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, struct_gate=None, mask=None,k_gate=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        # print('len_q: ', len_q)
        # print('len_k: ', len_k)
        # print('len_v: ', len_v)

        # if mask is not None:
        #     print('mask: ', mask.shape)

        if k_gate is not None:
            k_gate = k_gate.transpose(0, 1)
            k_gate=k_gate.reshape(n_head*sz_b,len_q,len_v)

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        #A Weighting score for the keys is provided
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        if mask is not None:
            # print('MHA mask.shape: ', mask.size())
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
            # print('MHA mask.shape after repeat: ', mask.size())
        output, attn = self.attention(q, k, v, mask=mask,k_gate=k_gate)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))

        output = self.layer_norm(output + residual)

        attn=attn.view(n_head,sz_b,len_q,len_v).transpose(0,1)
        
        return output, attn

# class SelectiveMultiHeadAttention(nn.Module):
#     ''' Selective Multi-Head Attention module '''

#     def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
#         super().__init__()

#         self.n_head = n_head
#         self.d_k = d_k
#         self.d_v = d_v

#         self.w_qs = nn.Linear(d_model, n_head * d_k)
#         self.w_ks = nn.Linear(d_model, n_head * d_k)
#         self.w_vs = nn.Linear(d_model, n_head * d_v)
#         nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
#         nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
#         nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

#         if self.selective:
#             self.attention = SelectiveScaledDotProductAttention(temperature=np.power(d_k, 0.5))
#         else:
#             self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
#         self.layer_norm = nn.LayerNorm(d_model)

#         self.fc = nn.Linear(n_head * d_v, d_model)
#         nn.init.xavier_normal_(self.fc.weight)

#         self.dropout = nn.Dropout(dropout)


#     def forward(self, q, k, v, output_gate, mask=None,k_gate=None):

#         d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

#         sz_b, len_q, _ = q.size()
#         sz_b, len_k, _ = k.size()
#         sz_b, len_v, _ = v.size()

#         if k_gate is not None:
#             k_gate = k_gate.transpose(0, 1)
#             k_gate=k_gate.reshape(n_head*sz_b,len_q,len_v)

#         residual = k

#         q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
#         k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
#         v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
#         #A Weighting score for the keys is provided
#         q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
#         k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
#         v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv
#         if mask is not None:
#             mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
#         output, attn = self.attention(q, k, v, mask=mask,k_gate=k_gate)

#         output = output.view(n_head, sz_b, len_k, d_v)
#         output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_k, -1) # b x lq x (n*dv)
        
#         output = self.dropout(self.fc(output))
#         output = self.layer_norm(output*output_gate + residual)

#         attn=attn.view(n_head,sz_b,len_q,len_k).transpose(0,1)
        
#         return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x
        output = self.w_2(F.relu(self.w_1(output)))
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


