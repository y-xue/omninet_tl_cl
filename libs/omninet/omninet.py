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

OmniNet API

"""
import torch
import os
import torch.nn as nn

from .peripherals import *
from .util import *
from .cnp import CNP

model_path = '/mnt-gluster/data/yxue/research/allstate/out'
# image_input_perph_pretrained_path = os.path.join(model_path, 'cifar2_res/cifar2_mbnt_mydl_0.01lr_decay_128bs300epochs.pt')
# video_input_perph_pretrained_path = os.path.join(model_path, 'hmdb2_res/hmdb2_convlstm_32hidden_channel_56max_pool_8bs50epochs.pt')
image_input_perph_pretrained_path = os.path.join(model_path, 'cifar10_res/cifar10_all_binary_mbnt_0.01lr128bs300epochs.pt')
video_input_perph_pretrained_path = os.path.join(model_path, 'hmdb_res/hmdb_binary_resnet_14x14_1024ic64hc3maxpool_dropout_0.3_0.5_input_dropout_0.5_nllloss_adam_0.0001lr32bs25epochs_best.pt')
english_language_perph_pretrained_path = os.path.join(model_path, 'transformer_clf_bertEmb/transformer_clf_bertEmb_mydl_2layer2head64d128dinner0.1dropout_0.0001lr64bs20epochs.pt')

model_path1 = '/files/yxue/research/allstate/out'
# image_input_perph_pretrained_path1 = os.path.join(model_path1, 'cifar2_res/cifar2_mbnt_mydl_0.01lr_decay_128bs300epochs.pt')
# video_input_perph_pretrained_path1 = os.path.join(model_path1, 'hmdb2_res/hmdb2_convlstm_32hidden_channel_56max_pool_8bs50epochs.pt')
image_input_perph_pretrained_path1 = os.path.join(model_path1, 'cifar10_res/cifar10_all_binary_mbnt_0.01lr128bs300epochs.pt')
video_input_perph_pretrained_path1 = os.path.join(model_path1, 'hmdb_res/hmdb_binary_resnet_14x14_1024ic64hc3maxpool_dropout_0.3_0.5_input_dropout_0.5_nllloss_adam_0.0001lr32bs25epochs_best.pt')
english_language_perph_pretrained_path1 = os.path.join(model_path1, 'transformer_clf_bertEmb/transformer_clf_bertEmb_mydl_2layer2head64d128dinner0.1dropout_0.0001lr64bs20epochs.pt')

class OmniNet(nn.Module):

    def __init__(self, config=None, gpu_id=-1,dropout=None,peripherals_type='default',unstructured_as_structured=False):
        super(OmniNet, self).__init__()
        if config is None:
            cc, pc, d = self.__defaultconf__()
        else:
            cc, pc, d = config
        if dropout is not None:
            cc['dropout']=dropout
            pc['dropout']=dropout
        self.gpu_id = gpu_id
        tasks = {'PENN': pc['penn_output_classes'], 'HMDB':pc['hmdb_output_classes'],
                 'IMAGE_CAPTION':pc['english_language_output_vocab'],'VQA':pc['vqa_output_vocab'],
                 'IMAGE_STRUCT_CLF':pc['image_struct_clf_output_classes'], 
                 'VQA_struct':pc['vqa_output_vocab'], 
                 'birds':pc['birds_output_classes'],'birds_struct':pc['birds_output_classes'],
                 'mm':pc['mm_output_classes'], 'mm_ITV':pc['mm_output_classes'], 'mm_vqa':pc['vqa_output_vocab'],
                 'SIQ':pc['SIQ_output_classes']
                 }
        # tasks = {'PENN': pc['penn_output_classes'], 'HMDB':pc['hmdb_output_classes'],
        #          'IMAGE_CAPTION':pc['english_language_output_vocab'],'VQA':pc['vqa_output_vocab']}
        self.cnp = CNP(tasks,conf=cc,domains=d, gpu_id=gpu_id)
        
        if peripherals_type == 'default':
            self.image_input_perph = ImageInputPeripheral(output_dim=cc['input_dim'],
                                                                      dropout=pc['dropout'],freeze_layers=not pc['unfreeze']['img'],
                                                                      pooling=cc['pooling'])
            self.english_language_perph = LanguagePeripheral(vocab_size=pc['english_language_input_vocab'],
                                                                         embed_dim=pc['english_language_input_embed'],
                                                                         output_dim=cc['input_dim'],
                                                                         lang='en',
                                                                         gpu_id=gpu_id,dropout=pc['dropout'])
            self.german_language_perph = LanguagePeripheral(vocab_size=pc['german_language_input_vocab'],
                                                                        embed_dim=pc['german_language_input_embed'],
                                                                        output_dim=cc['input_dim'],
                                                                        lang='de',
                                                                        gpu_id=gpu_id)
            self.audio_perph = FeaturePeripheral(74,cc['input_dim'],gpu_id=gpu_id)
            self.trs_perph = FeaturePeripheral(768,cc['input_dim'],gpu_id=gpu_id)
            self.video_input_perph = None
            self.struct_spat_perph = StructuredSpatialPeripheral(output_dim=cc['input_dim'],dropout=pc['struct_spat_dropout'],
                                            unstructured_as_structured=unstructured_as_structured,
                                            freeze_layers=not pc['unfreeze']['struct_spat'])
            self.struct_temp_perph = StructuredTemporalPeripheral(output_dim=cc['input_dim'],dropout=pc['struct_temp_dropout'],
                                            unstructured_as_structured=unstructured_as_structured,
                                            freeze_layers=not pc['unfreeze']['struct_temp'])
            self.struct_perph = StructuredPeripheral(output_dim=cc['logit_struct_periph_dim'],dropout=pc['struct_dropout'],
                                            unstructured_as_structured=unstructured_as_structured,
                                            freeze_layers=not pc['unfreeze']['struct_logits'])
        else:
            print('using timeline peripherals')

            try:
                self.image_input_perph = ImageInputPeripheralMobilenetV2(output_dim=cc['input_dim'],
                                                                          dropout=pc['dropout'],
                                                                          pretrained_path=image_input_perph_pretrained_path,
                                                                          freeze_layers=True)
                self.english_language_perph = LanguagePeripheralTSFM(output_dim=cc['input_dim'], embed_dim=64,
                                                                          dropout=pc['dropout'],
                                                                          pretrained_path=english_language_perph_pretrained_path)
                self.video_input_perph = VideoInputPeripheralConvLSTM(output_dim=cc['input_dim'],
                                                                          dropout=pc['dropout'],
                                                                          pretrained_path=video_input_perph_pretrained_path,
                                                                          freeze_layers=True)
            except:
                self.image_input_perph = ImageInputPeripheralMobilenetV2(output_dim=cc['input_dim'],
                                                                          dropout=pc['dropout'],
                                                                          pretrained_path=image_input_perph_pretrained_path1,
                                                                          freeze_layers=True)
                self.english_language_perph = LanguagePeripheralTSFM(output_dim=cc['input_dim'], embed_dim=64,
                                                                          dropout=pc['dropout'],
                                                                          pretrained_path=english_language_perph_pretrained_path1)
                self.video_input_perph = VideoInputPeripheralConvLSTM(output_dim=cc['input_dim'],
                                                                          dropout=pc['dropout'],
                                                                          pretrained_path=video_input_perph_pretrained_path1,
                                                                          freeze_layers=True)
        
    def reset(self,batch_size):
        self.cnp.reset(batch_size)

    # def perph_encode_videos(self,videos,domain='IMAGE'):
    #     return self.video_input_perph.encode(videos)

    # def perph_encode_images(self,images,domain='IMAGE'):
    #     return self.image_input_perph.encode(images)

    # def perph_encode_englishtexts(self,texts,domain='ENGLISH'):
    #     return sent_encodings,input_pad_mask=self.english_language_perph.embed_sentences(texts, tokenized=False)

    # def perph_encode_structured(self,structured):
    #     self.cnp.structured_emb(structured)

    def perph_encode_structured_saliency(self,structured_raw,structured_temp,structured_spat):
        self.cnp.structured_emb_saliency(structured_raw,structured_temp,structured_spat)
    
    # def encode_videos(self,video_encodings,domain='IMAGE'):
    #     self.cnp.encode(video_encodings,domain=domain)

    # def encode_images(self,image_encodings,domain='IMAGE'):
    #     self.cnp.encode(image_encodings,domain=domain)

    # def encode_englishtexts(self,sent_encodings,domain='ENGLISH'):
    #     self.cnp.encode(sent_encodings, pad_mask=input_pad_mask, domain=domain)


    def encode_videos(self,videos,domain='IMAGE'):
        # print('encode_videos')
        if self.video_input_perph is not None:
            video_encodings = self.video_input_perph.encode(videos)
            video_encodings_per_frame = None
        else:
            video_encodings, video_encodings_per_frame = self.image_input_perph.encode(videos)
        self.cnp.encode(video_encodings,domain=domain,features=video_encodings_per_frame)

    def encode_images(self,images,domain='IMAGE',bbox_mask=None,fusion=False):
        # print('encode_images')
        image_encodings, whole_image_encodings = self.image_input_perph.encode(images)
        # print('image_encodings:')
        # print(image_encodings[0])
        # print(image_encodings[20])
        # print(image_encodings[-1])
        if fusion:
            return self.cnp.encode_fusion(image_encodings,domain=domain,bbox_mask=bbox_mask,features=whole_image_encodings)
    
        return self.cnp.encode(image_encodings,domain=domain,bbox_mask=bbox_mask,features=whole_image_encodings)
    
    def encode_englishtexts(self,texts,domain='ENGLISH',fusion=False):
        # print('encode_englishtexts')
        sent_encodings,input_pad_mask=self.english_language_perph.embed_sentences(texts, tokenized=False)
        if fusion:
            return self.cnp.encode_fusion(sent_encodings, pad_mask=input_pad_mask, domain=domain)

        return self.cnp.encode(sent_encodings, pad_mask=input_pad_mask, domain=domain)

        # print('encode_englishtexts')
        # print(text.encode('ascii', errors='replace') for text in texts)
        # sent_encodings,input_pad_mask=self.english_language_perph.embed_sentences(texts)
        # print(sent_encodings,input_pad_mask)
        # print(sent_encodings.shape, input_pad_mask.shape)
        # b,t,s,f = sent_encodings.shape
        # if t > 1000:
        #     sent_encodings = sent_encodings[:,:1000,:,:]
        #     input_pad_mask = input_pad_mask[:,:1000]
            
        # return input_pad_mask

    def encode_trs(self, trs, domain='ENGLISH'):
        sent_encodings, input_pad_mask = self.trs_perph(trs)
        self.cnp.encode(sent_encodings, pad_mask=input_pad_mask, domain=domain)

    def encode_audios(self, audios, domain='AUDIO'):
        audio_encodings, input_pad_mask = self.audio_perph(audios)
        self.cnp.encode(audio_encodings, pad_mask=input_pad_mask, domain=domain)

    def encode_structured(self,structured,fusion=False):
        if self.cnp.inject_at_logits:
            if self.cnp.no_logit_struct_peripheral:
                self.cnp.structured_raw = structured
            else:
                self.cnp.structured_raw = self.struct_perph.encode(structured)

        if self.cnp.inject_at_encoder:

            if fusion:
                structured = structured.unsqueeze(1)
                # print('encode with struct peripherals')
                self.cnp.structured_spat = self.struct_spat_perph.encode(structured)
                self.cnp.structured_temp = self.struct_temp_perph.encode(structured)
            else:
                # print('1 structured:', structured.shape)
                structured = structured.unsqueeze(1).unsqueeze(1)

                # print('2 structured:', structured.shape)
                structured_spatial_enc = self.struct_spat_perph.encode(structured)

                # print('3 structured_spatial_enc:', structured_spatial_enc.shape)
                self.cnp.encode(structured_spatial_enc, domain='STRUCT_SPAT')

                structured_temporal_enc = self.struct_temp_perph.encode(structured)

                # print('4 structured_temporal_enc:', structured_temporal_enc.shape)
                self.cnp.encode(structured_temporal_enc, domain='STRUCT_TEMP')
                # self.cnp.fusion(structured, pad_mask=pad_mask)

    def encode_structured_saliency(self,structured_raw,structured_temp,structured_spat,fusion=False):
        self.cnp.structured_raw = structured_raw

        if self.cnp.inject_at_encoder:
            if fusion:
                structured_spat = structured_spat.unsqueeze(1)
                structured_temp = structured_temp.unsqueeze(1)
                
                self.cnp.structured_spat = self.struct_spat_perph.encode(structured_spat)
                self.cnp.structured_temp = self.struct_temp_perph.encode(structured_temp)
            else:
                structured_spat = structured_spat.unsqueeze(1).unsqueeze(1)
                structured_temp = structured_temp.unsqueeze(1).unsqueeze(1)

                structured_spatial_enc = self.struct_spat_perph.encode(structured_spat)
                self.cnp.encode(structured_spatial_enc, domain='STRUCT_SPAT')
                structured_temporal_enc = self.struct_temp_perph.encode(structured_temp)
                self.cnp.encode(structured_temporal_enc, domain='STRUCT_TEMP')

    def decode_from_targets(self,task,targets,target_pad_mask=None,return_attns=False):
        return self.cnp.decode(task, targets=targets,pad_mask=target_pad_mask,return_attns=return_attns)
    
    def decode_greedy(self,task, num_steps):
        return self.cnp.decode(task, targets=None, num_steps=num_steps)

    def save(self, checkpoint_dir, iterations, best_model=False):
        save_dir = os.path.join(checkpoint_dir, str(iterations))
        try:
            os.stat(save_dir)
        except:
            os.makedirs(save_dir)
        torch.save(self.state_dict(), os.path.join(save_dir, 'model.pth'))

        if best_model:
            torch.save(self.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
            print('Best model saved')
        print('Model saved, iterations: {}'.format(iterations))

    def restore(self, checkpoint_dir, iterations):
        save_dir = os.path.join(checkpoint_dir, str(iterations), 'model.pth')
        print(save_dir)
        pretrained_dict=torch.load(save_dir)
        model_dict=self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
        self.load_state_dict(pretrained_dict,strict=False)
        print('Restored existing model with iterations: {}'.format(iterations))
    
    def restore_file(self, file):
        pretrained_dict=torch.load(file)
        model_dict=self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
        self.load_state_dict(pretrained_dict,strict=False)
    
    @staticmethod
    def __defaultconf__():
        """
        The default confurigation as specified in the original paper

        """

        # timeline conf
        # cnp_conf = {
        #     'input_dim':64, #512,
        #     'control_dim':32,
        #     'output_dim':64, #512,
        #     'spatial_dim':64, #512,
        #     'temporal_dim':64, #512,
        #     'temporal_n_layers': 2, #6,
        #     'temporal_n_heads': 2, #8,
        #     'temporal_d_k':32, #64,
        #     'temporal_d_v':32, #64,
        #     'temporal_hidden_dim': 128, #2048,
        #     'decoder_dim': 64, #512,
        #     'decoder_n_layers': 2, #6,
        #     'decoder_n_heads': 2, #8,
        #     'decoder_d_k': 32, #64,
        #     'decoder_d_v': 32, #64,
        #     'decoder_hidden_dim': 128, #2048,
        #     'fusion_n_heads': 2, #8,
        #     'fusion_d_k': 32, #64,
        #     'fusion_d_v': 32, #64,
        #     'struct_dim':312,
        #     'max_seq_len':512,
        #     'output_embedding_dim': 50, #300,
        #     'dropout':0.1}
        # perph_conf = {
        #     'german_language_input_vocab': 25000,
        #     'german_language_input_embed': 300,
        #     'english_language_input_vocab': 25000,
        #     'english_language_input_embed': 300,
        #     'english_language_output_vocab': 25000,
        #     'german_language_output_vocab': 25000,
        #     'dropout': 0.1 ,
        #     'vqa_output_vocab':3500,
        #     'hmdb_output_classes':52,
        #     'penn_output_classes':48,
        #     'birds_output_classes':200,
        #     'mm_output_classes':2,
        #     'image_struct_clf_output_classes':2
        # }

        # domains = ['ENGLISH','GERMAN','IMAGE']

        cnp_conf = {
            'input_dim':512,
            'control_dim':32,
            'output_dim':512,
            'spatial_dim':512,
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
            'fusion_n_heads':8,
            'fusion_d_k':64,
            'fusion_d_v':64,
            'fusion_hidden_dim':2048,
            'struct_dim':312,
            'max_seq_len':500,
            'output_embedding_dim':300,
            'dropout':0.1,
            'use_s_decoder':False,
            'use_p_decoder':False}
        perph_conf = {
            'german_language_input_vocab': 25000,
            'german_language_input_embed': 300,
            'english_language_input_vocab': 25000,
            'english_language_input_embed': 300,
            'english_language_output_vocab': 25000,
            'german_language_output_vocab': 25000,
            'dropout': 0.1 ,
            'vqa_output_vocab':3500,
            'hmdb_output_classes':52,
            'penn_output_classes':48,
            'birds_output_classes':200,
            'mm_output_classes':2,
            'image_struct_clf_output_classes':2
        }

        domains = ['ENGLISH','GERMAN','IMAGE']#,'STRUCT']

        return cnp_conf, perph_conf, domains
