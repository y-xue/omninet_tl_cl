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

OmniNet temporal Encoder and spatio-temporal decoder layer

"""
import torch.nn as nn
import torch
from libs.omninet.cnp.SubLayers import MultiHeadAttention, PositionwiseFeedForward, MultiHeadFusionAttention


class FusionLayer(nn.Module):
    # for now, d_inner is not used because there is no PositionwiseFeedForward layer
    def __init__(self, d_model, n_head, d_k, d_v, d_inner=None, dropout=0.1, attention_type='default', convex_gate=False):
        super(FusionLayer, self).__init__()

        if attention_type == 'default':
            self.fusion_attn = MultiHeadAttention(
                n_head, d_model, d_k, d_v, dropout=dropout)
        else:
            self.fusion_attn = MultiHeadFusionAttention(
                n_head, d_model, d_k, d_v, dropout=dropout, 
                attention_type=attention_type, convex_gate=convex_gate)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, structured, cache, non_pad_mask, struct_gate=None, fusion_attn_mask=None):
        # structured = torch.empty(128, 512, 512).uniform_(0,1) # sz_b, len_q, d_model
        # print(cache.size())
        # print('FusionLayer forward struct_gate:', struct_gate)
        fusion_output, fusion_attn_score = self.fusion_attn(
            structured, cache, structured, struct_gate, mask=fusion_attn_mask)
        if non_pad_mask is not None: fusion_output*=non_pad_mask
        fusion_output = self.pos_ffn(fusion_output)
        if non_pad_mask is not None: fusion_output*=non_pad_mask
        return fusion_output, fusion_attn_score

# class GateLayer(nn.Module):
#     def __init__(self, d_inner, n_head, d_k, d_v, cache_dim, dropout=0.1, gpu_id=-1):
#         super(DecoderLayer, self).__init__()
#         self.gpu_id=gpu_id
#         self.struct_cache_attn = MultiHeadAttention(n_head, cache_dim, d_k, d_v, dropout=dropout)
#         self.pos_ffn = PositionwiseFeedForward(cache_dim, d_inner, dropout=dropout)

#     def forward(self, structured, cache,struct_enc_attn_mask=None):
        
#         dec_output, dec_temp_attn = self.struct_cache_attn(
#             structured, cache, cache, mask=struct_enc_attn_mask)
        
#         dec_output = self.pos_ffn(dec_output)
#         return dec_output,[dec_slf_attn,dec_spat_attn,dec_temp_attn]

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input,non_pad_mask,slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        if non_pad_mask is not None: enc_output*=non_pad_mask
        enc_output = self.pos_ffn(enc_output)
        if non_pad_mask is not None: enc_output*=non_pad_mask
        return enc_output, enc_slf_attn



class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, temporal_dim, spatial_dim,dropout=0.1,gpu_id=-1):
        super(DecoderLayer, self).__init__()
        self.gpu_id=gpu_id
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.temporal_cache_attn = MultiHeadAttention(n_head, temporal_dim, d_k, d_v, dropout=dropout)
        self.temporal_proj=nn.Linear(d_model,temporal_dim)
        self.spatial_proj=nn.Linear(temporal_dim,spatial_dim)
        self.spatial_cache_attn = MultiHeadAttention(n_head, spatial_dim, d_k, d_v, dropout=dropout)
        self.spat_dec_proj = nn.Linear(spatial_dim,d_model)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)


    def forward(self, dec_input,temporal_cache, spatial_cache,temporal_spatial_link,non_pad_mask,slf_attn_mask=None,dec_enc_attn_mask=None):
        #First attend the output encodings on itself
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        if non_pad_mask is not None: dec_output*=non_pad_mask
        #Attend hidden states on the temporal cache
        dec_temp=self.temporal_proj(dec_output)

        dec_temp, dec_temp_attn = self.temporal_cache_attn(
            dec_temp, temporal_cache, temporal_cache, mask=dec_enc_attn_mask)
        if non_pad_mask is not None: dec_temp*=non_pad_mask
        # Attend hidden states on the spatial cache
        dec_spat=self.spatial_proj(dec_temp)
        dec_spat_attn=None
        if spatial_cache is not None:
            # Process the spatial cache and add the respective weightings
            spatial_gate = []
            idx_start = 0
            # print('dec_temp_attn:', dec_temp_attn.shape)
            for l in temporal_spatial_link:
                # print('temporal_spatial_link:', l)
                t, s = l
                if s == 0 and t == 1:
                    # STRUCT_TEMP
                    continue

                if s > 1 or (s == 1 and t == 0):
                    # STRUCT_SPAT
                    if t == 0:
                        t = 1
                    temp_sel = dec_temp_attn[:, :, :, idx_start:idx_start + t]
                    b, nh, dq, t = temp_sel.shape # (b, nh, N, t)
                    # print('0 temp_sel:', temp_sel.shape)
                    temp_sel = temp_sel.unsqueeze(4).expand(b, nh, dq, t, s).transpose(3, 4)
                    # print('1 temp_sel:', temp_sel.shape)
                    temp_sel = temp_sel.reshape(b, nh, dq, t * s)
                    # print('2 temp_sel:', temp_sel.shape)
                    spatial_gate.append(temp_sel)
                idx_start = idx_start + t
                # print(idx_start)
            spatial_gate = torch.cat(spatial_gate, dim=3)
            dec_spat,dec_spat_attn=self.spatial_cache_attn(dec_spat,spatial_cache,spatial_cache,
                                                           k_gate=spatial_gate)
            if non_pad_mask is not None: dec_spat*=non_pad_mask
               
        dec_output=self.spat_dec_proj(dec_spat)
        dec_output = self.pos_ffn(dec_output)
        if non_pad_mask is not None: dec_output*=non_pad_mask
        return dec_output,[dec_slf_attn,dec_spat_attn,dec_temp_attn]

class DecoderLayerSingleCache(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, cache_dim, dropout=0.1, gpu_id=-1):
        super(DecoderLayerSingleCache, self).__init__()
        self.gpu_id=gpu_id
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.cache_attn = MultiHeadAttention(n_head, cache_dim, d_k, d_v, dropout=dropout)
        self.cache_proj=nn.Linear(d_model,cache_dim)
        self.dec_proj = nn.Linear(cache_dim,d_model)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)


    def forward(self, dec_input,cache,non_pad_mask,slf_attn_mask=None,dec_enc_attn_mask=None):
        #First attend the output encodings on itself
        # print('dec_input: ', dec_input.shape)
        # print('cache: ', cache.shape)
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input,mask=slf_attn_mask)
        if non_pad_mask is not None: dec_output*=non_pad_mask

        # print('dec_output: ', dec_output.shape)
        #Attend hidden states on the temporal cache
        dec_output=self.cache_proj(dec_output)
        # print('dec_output: ', dec_output.shape)

        dec_output, dec_cache_attn = self.cache_attn(
            dec_output, cache, cache,mask=dec_enc_attn_mask)
        if non_pad_mask is not None: dec_output*=non_pad_mask
        
        dec_output = self.dec_proj(dec_output)
        dec_output = self.pos_ffn(dec_output)
        if non_pad_mask is not None: dec_output*=non_pad_mask
        return dec_output,[dec_slf_attn,dec_cache_attn]
