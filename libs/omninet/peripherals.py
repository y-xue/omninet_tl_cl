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

OmniNet standard input peripherals

"""

# yolov4_source = '/pytorch-YOLOv4'

# import sys
# sys.path.insert(1,yolov4_source)
# from tool.utils import *
# from tool.torch_utils import *
# from tool.darknet2pytorch import Darknet

from bpemb import BPEmb
from torch.nn.functional import relu, log_softmax
from .base_models.resnet import resnet50, resnet152
from .base_models.mobilenet_v2 import MobileNetV2_Cifar10
from .base_models.conv_lstm import ConvLSTM_HMDB
from .base_models.transformer_bertEmb import TransformerBertEmb_IMDB
from .base_models.linear import FFN
from .util import *

from pytorch_transformers import BertTokenizer, BertModel
import cv2


TSFM_NAME = '/mnt-gluster/data/yxue/research/transformers/bert-base-uncased'
TSFM_NAME1 = '/files/yxue/research/transformers/bert-base-uncased'
bpemb_pretrained_path3 = '/mnt-gluster/data/yxue/research/omninet_data/bpemb'
bpemb_pretrained_path2 = '/files/yxue/research/bpemb'
bpemb_pretrained_path1 = '/scratch1/yxuea/data/models/bpemb'

yolov4_weights = '/files/yxue/research/yolov4.weights'

structured_model1 = '/scratch1/yxuea/data/struct_peripheral.pth'
structured_model2 = '/files/yxue/research/allstate/out/omninet/structured_clustering_std3/best_model.pth'

class base_peripheral(nn.Module):
    """
        The base standard non recursive perpheral
        All base peripherals must implement the following functions:
            __init__()
            run_cycle()

    """
    def __init__(self):
        super(base_peripheral,self).__init__()


##############################################################
##############################################################
# Definition of standard peripherals for most common tasks   #
##############################################################

class ImageInputPeripheralMobilenetV2(base_peripheral):
    def __init__(self,output_dim,dropout=0,pretrained_path='/mnt-gluster/data/yxue/research/allstate',freeze_layers=True):
        self.feature_dim=1280 
        super(ImageInputPeripheralMobilenetV2,self).__init__()
        self.image_model=MobileNetV2_Cifar10(pretrained=pretrained_path)
        # self.image_model = self.image_model.cuda(0)
        if freeze_layers:
            self.image_model=self.image_model.eval()
            #Override the train mode So that it does not change when switching OmniNet to train mode
            self.image_model.train=self.empty_fun
            self.image_model.eval=self.empty_fun
            for param in self.image_model.parameters():
                param.requires_grad = False
        self.enc_dropout=nn.Dropout(dropout)   
        self.output_fc=nn.Linear(self.feature_dim,output_dim)

    def encode(self,image_tensor):
        shape=image_tensor.shape
        if len(shape)==5:
            t_dim=image_tensor.shape[1]
            image_tensor=torch.reshape(image_tensor,(-1,3,shape[3],shape[4]))    
        batch_size=image_tensor.shape[0]
        image_enc=self.image_model(image_tensor)
        # image_enc: [b, 1280, 4, 4]

        enc_reshape=torch.reshape(image_enc,[batch_size,self.feature_dim,-1])
        enc_transposed=torch.transpose(enc_reshape,1,2)
        drp_enc=self.enc_dropout(enc_transposed)
        output_enc=self.output_fc(drp_enc)
        if len(shape)==5:
            output_enc=torch.reshape(output_enc,(-1,t_dim,output_enc.shape[1],output_enc.shape[2]))
        else:
            output_enc=output_enc.unsqueeze(1) # b,t,s,f
        return output_enc, None

    def empty_fun(self,mode):
        pass

class LanguagePeripheralTSFM(base_peripheral):
    def __init__(self,output_dim,embed_dim,pretrained_path='/mnt-gluster/data/yxue/research/allstate',gpu_id=-1,dropout=0):
        super(LanguagePeripheralTSFM,self).__init__()
        self.gpu_id=gpu_id

        try:
            self.tokenizer = BertTokenizer.from_pretrained(TSFM_NAME1)
            bert = BertModel.from_pretrained(TSFM_NAME1)
        except:
            self.tokenizer = BertTokenizer.from_pretrained(TSFM_NAME)
            bert = BertModel.from_pretrained(TSFM_NAME)

        self.pad_char = self.tokenizer.vocab['[PAD]']
        self.encoder = TransformerBertEmb_IMDB(pretrained_path, bert, pad_id=self.tokenizer.pad_token_id)
        # self.encoder = self.encoder.cuda(0)

        # self.encoder = TransformerBertEmb(2, bert, d_word_vec, n_layers, n_head, d_model, d_inner, dropout=dropout, n_position=n_position, pad_id=tokenizer.pad_token_id)
        # self.encoder.load_state_dict(pretrained_path)

        self.encoder=self.encoder.eval()
        #Override the train mode So that it does not change when switching OmniNet to train mode
        self.encoder.train=self.empty_fun
        self.encoder.eval=self.empty_fun
        for param in self.encoder.parameters():
            param.requires_grad = False

        # # Add an extra padding character
        # self.embed_layer=nn.Embedding(vocab_size+1,embed_dim,padding_idx=self.pad_char)
        # if(embedding_preload==True):
        #     self.embed_layer.load_state_dict({'weight': torch.tensor(self.bpe_encoder.emb.vectors)})
        #     print("Loading pretrained word embeddings.")
        self.enc_dropout = nn.Dropout(dropout)
        self.output=nn.Linear(embed_dim,output_dim)
        
    def forward(self,tokens):        
        pad_mask = tokens.eq(self.id_PAD)     # tokens: n_sentences * max_len
        embeddings, _ = self.encoder(tokens) # embeddings: n_sentences * max_len * embed_dim
        embeddings = self.enc_dropout(embeddings)
        output = self.output(embeddings)      # output: n_sentences * max_len * output_dim (projected)
        return output.unsqueeze(2)          # output unsqueeze: n_sentences * max_len * 1 * output_dim

    def embed_sentences(self,sentences,tokenized=True):
        # Generate the tokens using BPEmb
        if tokenized:
            pad_mask = sentences.eq(self.id_PAD) #[sent.eq(self.id_PAD) for sent in sentences]
            tokens = sentences
        else:
            tokens,pad_mask=self.tokenize_sentences(sentences)
        
        return self.forward(tokens),pad_mask

    def decode_tokens(self,tokens):
        if isinstance(tokens,torch.Tensor):
            tokens=tokens.cpu().numpy().astype(int).tolist()
        elif isinstance(tokens,np.ndarray):
            tokens=tokens.astype(int).tolist()
        #Filter out all tokens which have values larger than vocab_size and filter all elements after EOS
        filtered_tokens=[]
        for t in tokens:
            values=[]
            for i in t:
                if i==self.id_EOS:
                    break
                elif i<self.id_PAD:
                    values.append(i)
            filtered_tokens.append(values)
        #Remove all the padding characters in a list
        return self.tokenizer.decode(filtered_tokens)

    def tokenize_sentences(self, sentences, max_len=512, PAD='[PAD]', SEP='[SEP]', CLS='[CLS]'):
        sentence_tokens = []
        for text in sentences:
            tokens = self.tokenizer.tokenize(text)
            tokens = tokens[:max_len-2]
            pad = [self.tokenizer.vocab[PAD]] * (max_len-len(tokens)-2)
            indexed_tokens = [self.tokenizer.vocab[CLS]] + self.tokenizer.convert_tokens_to_ids(tokens) + [self.tokenizer.vocab[SEP]] + pad
            sentence_tokens.append(indexed_tokens)

        sentence_tokens = torch.tensor(np.array(sentence_tokens)).cuda(0)
        pad_mask = sentence_tokens.eq(self.id_PAD)
        return sentence_tokens, pad_mask

    # def tokenize_sentences(self,sentences,max_sent_len=500,truncating='pre'):
    #     tokens = self.bpe_encoder.encode_ids_with_bos_eos(sentences)
    #     # Pad the tokens with the pad_char
    #     max_len = 0
        
    #     for i in range(len(tokens)):
    #         if len(tokens[i]) - 2 > max_sent_len:
    #             if truncating == 'pre':
    #                 tokens[i] = tokens[i][:1] + tokens[i][-(max_sent_len-1):]
    #             else:
    #                 tokens[i] = tokens[i][:(max_sent_len-1)] + tokens[i][-1:]
    #         max_len = max(max_len, len(tokens[i]))
    #     for i in range(len(tokens)):
    #         tok_len = len(tokens[i])
    #         tokens[i].extend([self.pad_char]*(max_len-tok_len))
    #     tokens = torch.tensor(np.array(tokens)) # n_sentences * max_len
    #     if self.gpu_id > -1:
    #         tokens = tokens.cuda(self.gpu_id)
    #     pad_mask=tokens.eq(self.id_PAD)
    #     return tokens,pad_mask

    @property
    def id_PAD(self):
        return self.pad_char

    # @property
    # def id_GO(self):
    #     return 1

    @property
    def id_EOS(self):
        return self.tokenizer.vocab['[SEP]']

    def empty_fun(self,mode):
        pass

class VideoInputPeripheralConvLSTM(base_peripheral):
    def __init__(self,output_dim,dropout=0,pretrained_path='/mnt-gluster/data/yxue/research/allstate',freeze_layers=True):
        self.feature_dim=64 # fd
        super(VideoInputPeripheralConvLSTM,self).__init__()
        # self.model=ConvLSTM(input_channels=3, hidden_channels=[32], kernel_size=3, step=16,
        #                 effective_step=list(range(16)))
        self.model = ConvLSTM_HMDB(pretrained_path)
        # self.model = self.model.cuda(0)

        self.model=self.model.eval()
        #Override the train mode So that it does not change when switching OmniNet to train mode
        self.model.train=self.empty_fun
        self.model.eval=self.empty_fun
        for param in self.model.parameters():
            param.requires_grad = False

        self.enc_dropout=nn.Dropout(dropout)
        self.output_fc=nn.Linear(self.feature_dim,output_dim)

    def encode(self, video):
        shape=video.shape # [b, seq, c, h, w]
        batch_size=shape[0]
        n_seq = shape[1]
        # print(video.shape)
        
        video_enc=torch.stack(self.model(video)).permute(1,0,2,3,4) # [b, seq, c, h, w] 
        # video_enc: [b, 16, 32, 4, 4]

        enc_reshape=torch.reshape(video_enc,[batch_size,n_seq,self.feature_dim,-1]) # [b, seq, fd, s]
        enc_transposed=torch.transpose(enc_reshape,2,3) # [b, seq, s, fd]
        drp_enc=self.enc_dropout(enc_transposed)
        output_enc=self.output_fc(drp_enc) # [b, seq, s, f]
        
        return output_enc

    def empty_fun(self,mode):
        pass


class ImageInputPeripheral(base_peripheral):
    def __init__(self,output_dim,dropout=0,weights_preload=True,freeze_layers=True,pooling=False):
        print('ImageInputPeripheral freeze_layers:', freeze_layers)
        self.pooling = pooling
        self.feature_dim=2048  
        super(ImageInputPeripheral,self).__init__()
        self.image_model=resnet152(pretrained=weights_preload)
        if freeze_layers:
            self.image_model=self.image_model.eval()
            #Override the train mode So that it does not change when switching OmniNet to train mode
            self.image_model.train=self.empty_fun
            self.image_model.eval=self.empty_fun
            for param in self.image_model.parameters():
                param.requires_grad = False
        self.enc_dropout=nn.Dropout(dropout)   
        self.output_fc=nn.Linear(self.feature_dim,output_dim)

        if pooling:
            self.avgpool=nn.AdaptiveAvgPool2d((1, 1))
            self.enc_fc=nn.Linear(self.feature_dim,output_dim)

    def encode(self,image_tensor):
        # print('image peripheral:')
        # print('image_tensor[0] and [-1]:')
        # print(image_tensor[0])
        # print(image_tensor[-1])
        shape=image_tensor.shape
        if len(shape)==5:
            t_dim=image_tensor.shape[1]
            image_tensor=torch.reshape(image_tensor,(-1,3,shape[3],shape[4]))    
        batch_size=image_tensor.shape[0]
        image_enc=self.image_model(image_tensor)

        # print('image_enc:')
        # print(image_enc[0])
        # print(image_enc[-1])
        enc_reshape=torch.reshape(image_enc,[batch_size,self.feature_dim,-1])
        enc_transposed=torch.transpose(enc_reshape,1,2)
        drp_enc=self.enc_dropout(enc_transposed)
        output_enc=self.output_fc(drp_enc)
        # print('output_enc:')
        # print(output_enc[0])
        # print(output_enc[-1])
        if len(shape)==5:
            output_enc=torch.reshape(output_enc,(-1,t_dim,output_enc.shape[1],output_enc.shape[2]))
        else:
            output_enc=output_enc.unsqueeze(1)

        if self.pooling:
            # print('image_enc:', image_enc.shape)
            whole_image_enc = self.avgpool(image_enc)
            # print('whole_image_enc:',whole_image_enc.shape)
            whole_image_enc = self.enc_fc(whole_image_enc.squeeze())
            return output_enc, whole_image_enc

        return output_enc, None

    def empty_fun(self,mode):
        pass



class LanguagePeripheral(base_peripheral):
    def __init__(self,output_dim,vocab_size=10000,embed_dim=50,lang='en',embedding_preload=True,gpu_id=-1,dropout=0):
        super(LanguagePeripheral,self).__init__()
        self.gpu_id=gpu_id
        self.pad_char = vocab_size
        
        if os.path.exists(bpemb_pretrained_path1):
            bpath = bpemb_pretrained_path1
        elif os.path.exists(bpemb_pretrained_path2):
            bpath = bpemb_pretrained_path2
        elif os.path.exists(bpemb_pretrained_path3):
            bpath = bpemb_pretrained_path3
        else:
            bpath = None

        self.bpe_encoder=BPEmb(lang=lang, vs=vocab_size,dim=embed_dim,add_pad_emb=True,cache_dir=bpath)
        
        # except:
        #     try:
        #         self.bpe_encoder=BPEmb(lang=lang, vs=vocab_size,dim=embed_dim,add_pad_emb=True,cache_dir=bpemb_pretrained_path)
        #     except:
        #         self.bpe_encoder=BPEmb(lang=lang, vs=vocab_size,dim=embed_dim,add_pad_emb=True)#,cache_dir='/mnt-gluster/data/yxue/research/omninet_data/bpemb')
        # Add an extra padding character
        self.embed_layer=nn.Embedding(vocab_size+1,embed_dim,padding_idx=self.pad_char)
        if(embedding_preload==True):
            self.embed_layer.load_state_dict({'weight': torch.tensor(self.bpe_encoder.emb.vectors)})
            print("Loading pretrained word embeddings.")
        self.enc_dropout = nn.Dropout(dropout)
        self.output=nn.Linear(embed_dim,output_dim)
        
    def forward(self,tokens):        
        pad_mask=tokens.eq(self.id_PAD)
        embeddings=self.embed_layer(tokens)
        embeddings=self.enc_dropout(embeddings)
        output=self.output(embeddings)
        return output.unsqueeze(2)

    def embed_sentences(self,sentences,tokenized=False):
        # Generate the tokens using BPEmb
        tokens,pad_mask=self.tokenize_sentences(sentences)
        return self.forward(tokens),pad_mask

    def decode_tokens(self,tokens):
        if isinstance(tokens,torch.Tensor):
            tokens=tokens.cpu().numpy().astype(int).tolist()
        elif isinstance(tokens,np.ndarray):
            tokens=tokens.astype(int).tolist()
        #Filter out all tokens which have values larger than vocab_size and filter all elements after EOS
        filtered_tokens=[]
        for t in tokens:
            values=[]
            for i in t:
                if i==self.id_EOS:
                    break
                elif i<self.id_PAD:
                    values.append(i)
            filtered_tokens.append(values)
        #Remove all the padding characters in a list
        return self.bpe_encoder.decode_ids(filtered_tokens)


    def tokenize_sentences(self,sentences):
        tokens = self.bpe_encoder.encode_ids_with_bos_eos(sentences)
        # Pad the tokens with the pad_char
        max_len = 0
        
        for t in tokens:
            max_len = max(max_len, len(t))
        for i in range(len(tokens)):
            tok_len = len(tokens[i])
            tokens[i].extend([self.pad_char]*(max_len-tok_len))
        tokens = torch.tensor(np.array(tokens))
        if self.gpu_id > -1:
            tokens = tokens.cuda(self.gpu_id)
        pad_mask=tokens.eq(self.id_PAD)
        return tokens,pad_mask

    @property
    def id_PAD(self):
        return self.pad_char

    @property
    def id_GO(self):
        return 1

    @property
    def id_EOS(self):
        return 2

class StructuredPeripheral(base_peripheral):
    def __init__(self,output_dim,dropout=0,freeze_layers=True,unstructured_as_structured=False):
        super(StructuredPeripheral,self).__init__()
        print('StructuredPeripheral freeze_layers', freeze_layers)

        self.unstructured_as_structured = unstructured_as_structured
        feature_dim = 512
        self.model = FFN(128, 3500, feature_dim)

        if os.path.exists(structured_model1):
            structured_model = structured_model1
        else:
            structured_model = structured_model2
        self.model.load_state_dict(torch.load(structured_model))

        if freeze_layers:
            self.model=self.model.eval()
            #Override the train mode So that it does not change when switching OmniNet to train mode
            self.model.train=self.empty_fun
            self.model.eval=self.empty_fun
            for param in self.model.parameters():
                param.requires_grad = False

        self.enc_dropout=nn.Dropout(dropout)   
        self.output_fc=nn.Linear(feature_dim,output_dim)

        print('StructuredPeripheral dropout:', dropout)

    def encode(self, s):
        if self.unstructured_as_structured:
            output_enc=self.output_fc(s)
        else:
            f = self.model(s)
            drp_enc=self.enc_dropout(f)
            output_enc=self.output_fc(drp_enc)
        
        return output_enc

    def empty_fun(self,mode):
        pass

class StructuredSpatialPeripheral(base_peripheral):
    def __init__(self,output_dim,dropout=0,freeze_layers=True,unstructured_as_structured=False):
        super(StructuredSpatialPeripheral,self).__init__()

        print('StructuredSpatialPeripheral freeze_layers', freeze_layers)

        self.unstructured_as_structured = unstructured_as_structured
        feature_dim = 512
        self.model = FFN(128, 3500, feature_dim)

        if os.path.exists(structured_model1):
            structured_model = structured_model1
        else:
            structured_model = structured_model2
        self.model.load_state_dict(torch.load(structured_model))

        if freeze_layers:
            self.model=self.model.eval()
            #Override the train mode So that it does not change when switching OmniNet to train mode
            self.model.train=self.empty_fun
            self.model.eval=self.empty_fun
            for param in self.model.parameters():
                param.requires_grad = False

        self.enc_dropout=nn.Dropout(dropout)   
        self.output_fc=nn.Linear(feature_dim,output_dim)

        print('StructuredSpatialPeripheral dropout:', dropout)

    def encode(self, s):
        if self.unstructured_as_structured:
            output_enc=self.output_fc(s)
        else:
            f = self.model(s)
            drp_enc=self.enc_dropout(f)
            output_enc=self.output_fc(drp_enc)
        return output_enc

    def empty_fun(self,mode):
        pass

class StructuredTemporalPeripheral(base_peripheral):
    def __init__(self,output_dim,dropout=0,freeze_layers=True,unstructured_as_structured=False):
        super(StructuredTemporalPeripheral,self).__init__()

        print('StructuredTemporalPeripheral freeze_layers', freeze_layers)

        self.unstructured_as_structured = unstructured_as_structured
        feature_dim = 512
        self.model = FFN(128, 3500, feature_dim)

        if os.path.exists(structured_model1):
            structured_model = structured_model1
        else:
            structured_model = structured_model2
        self.model.load_state_dict(torch.load(structured_model))

        if freeze_layers:
            self.model=self.model.eval()
            #Override the train mode So that it does not change when switching OmniNet to train mode
            self.model.train=self.empty_fun
            self.model.eval=self.empty_fun
            for param in self.model.parameters():
                param.requires_grad = False

        self.enc_dropout=nn.Dropout(dropout)   
        self.output_fc=nn.Linear(feature_dim,output_dim)
        print('StructuredSpatialPeripheral dropout:', dropout)

    def encode(self, s):
        if self.unstructured_as_structured:
            output_enc=self.output_fc(s)
        else:
            f = self.model(s)
            drp_enc=self.enc_dropout(f)
            output_enc=self.output_fc(drp_enc)
        return output_enc

    def empty_fun(self,mode):
        pass

# class ImageBoundingBoxPeripheral(base_peripheral):
#     def __init__(self, freeze_layers=True): 
#         super(ImageBoundingBoxPeripheral,self).__init__()
#         self.model = Darknet(yolov4_source + '/cfg/yolov4.cfg')

#         m.load_weights(yolov4_weights)

#         if freeze_layers:
#             self.model=self.model.eval()
#             #Override the train mode So that it does not change when switching OmniNet to train mode
#             self.model.train=self.empty_fun
#             self.model.eval=self.empty_fun
#             for param in self.model.parameters():
#                 param.requires_grad = False
        
#     def encode(self,image_tensor):
#         # print('image bb peripheral:')
#         # print('image_tensor[0] and [-1]:')
#         # print(image_tensor[0])
#         # print(image_tensor[-1])
#         shape=image_tensor.shape
#         if len(shape)==5:
#             t_dim=image_tensor.shape[1]
#             image_tensor=torch.reshape(image_tensor,(-1,3,shape[3],shape[4]))    
#         batch_size=image_tensor.shape[0]
#         out=self.model(image_tensor)

#         bbs=torch.stack([out[x]['boxes'] for x in range(len(out))])

#         if len(shape)==5:
#             output_enc=torch.reshape(output_enc,(-1,t_dim,output_enc.shape[1],output_enc.shape[2]))
#         else:
#             output_enc=output_enc.unsqueeze(1)
#         return output_enc

#     def empty_fun(self,mode):
#         pass

# class ImageInputPeripheral(base_peripheral):
#     def __init__(self,output_dim,dropout=0,weights_preload=True,freeze_layers=True):
#         self.feature_dim=2048  
#         super(ImageInputPeripheral,self).__init__()
#         self.image_model=resnet152(pretrained=weights_preload)
#         if freeze_layers:
#             self.image_model=self.image_model.eval()
#             #Override the train mode So that it does not change when switching OmniNet to train mode
#             self.image_model.train=self.empty_fun
#             self.image_model.eval=self.empty_fun
#             for param in self.image_model.parameters():
#                 param.requires_grad = False
#         self.enc_dropout=nn.Dropout(dropout)   
#         self.output_fc=nn.Linear(self.feature_dim,output_dim)

#     def encode(self,image_tensor):
#         shape=image_tensor.shape
#         if len(shape)==5:
#             t_dim=image_tensor.shape[1]
#             image_tensor=torch.reshape(image_tensor,(-1,3,shape[3],shape[4]))    
#         batch_size=image_tensor.shape[0]
#         image_enc=self.image_model(image_tensor)
#         enc_reshape=torch.reshape(image_enc,[batch_size,self.feature_dim,-1])
#         enc_transposed=torch.transpose(enc_reshape,1,2)
#         drp_enc=self.enc_dropout(enc_transposed)
#         output_enc=self.output_fc(drp_enc)
#         if len(shape)==5:
#             output_enc=torch.reshape(output_enc,(-1,t_dim,output_enc.shape[1],output_enc.shape[2]))
#         else:
#             output_enc=output_enc.unsqueeze(1) # b,t,s,f
#         return output_enc

#     def empty_fun(self,mode):
#         pass



# class LanguagePeripheral(base_peripheral):
#     def __init__(self,output_dim,vocab_size=10000,embed_dim=50,lang='en',embedding_preload=True,gpu_id=-1,dropout=0):
#         super(LanguagePeripheral,self).__init__()
#         self.gpu_id=gpu_id
#         self.pad_char = vocab_size
#         try:
#             self.bpe_encoder=BPEmb(lang=lang, vs=vocab_size,dim=embed_dim,add_pad_emb=True,cache_dir='/mnt-gluster/data/yxue/research/omninet_data/bpemb')
#         except:
#             self.bpe_encoder=BPEmb(lang=lang, vs=vocab_size,dim=embed_dim,add_pad_emb=True)#,cache_dir='/mnt-gluster/data/yxue/research/omninet_data/bpemb')
#         # Add an extra padding character
#         self.embed_layer=nn.Embedding(vocab_size+1,embed_dim,padding_idx=self.pad_char)
#         if(embedding_preload==True):
#             self.embed_layer.load_state_dict({'weight': torch.tensor(self.bpe_encoder.emb.vectors)})
#             print("Loading pretrained word embeddings.")
#         self.enc_dropout = nn.Dropout(dropout)
#         self.output=nn.Linear(embed_dim,output_dim)
        
#     def forward(self,tokens):        
#         pad_mask=tokens.eq(self.id_PAD)     # tokens: n_sentences * max_len
#         embeddings=self.embed_layer(tokens) # embeddings: n_sentences * max_len * embed_dim
#         embeddings=self.enc_dropout(embeddings)
#         output=self.output(embeddings)      # output: n_sentences * max_len * output_dim (projected)
#         return output.unsqueeze(2)          # output unsqueeze: n_sentences * max_len * 1 * output_dim

#     def embed_sentences(self,sentences):
#         # Generate the tokens using BPEmb
#         tokens,pad_mask=self.tokenize_sentences(sentences)
#         return self.forward(tokens),pad_mask

#     def decode_tokens(self,tokens):
#         if isinstance(tokens,torch.Tensor):
#             tokens=tokens.cpu().numpy().astype(int).tolist()
#         elif isinstance(tokens,np.ndarray):
#             tokens=tokens.astype(int).tolist()
#         #Filter out all tokens which have values larger than vocab_size and filter all elements after EOS
#         filtered_tokens=[]
#         for t in tokens:
#             values=[]
#             for i in t:
#                 if i==self.id_EOS:
#                     break
#                 elif i<self.id_PAD:
#                     values.append(i)
#             filtered_tokens.append(values)
#         #Remove all the padding characters in a list
#         return self.bpe_encoder.decode_ids(filtered_tokens)


#     def tokenize_sentences(self,sentences,max_sent_len=500,truncating='pre'):
#         tokens = self.bpe_encoder.encode_ids_with_bos_eos(sentences)
#         # Pad the tokens with the pad_char
#         max_len = 0
        
#         for i in range(len(tokens)):
#             if len(tokens[i]) - 2 > max_sent_len:
#                 if truncating == 'pre':
#                     tokens[i] = tokens[i][:1] + tokens[i][-(max_sent_len-1):]
#                 else:
#                     tokens[i] = tokens[i][:(max_sent_len-1)] + tokens[i][-1:]
#             max_len = max(max_len, len(tokens[i]))
#         for i in range(len(tokens)):
#             tok_len = len(tokens[i])
#             tokens[i].extend([self.pad_char]*(max_len-tok_len))
#         tokens = torch.tensor(np.array(tokens)) # n_sentences * max_len
#         if self.gpu_id > -1:
#             tokens = tokens.cuda(self.gpu_id)
#         pad_mask=tokens.eq(self.id_PAD)
#         return tokens,pad_mask

#     @property
#     def id_PAD(self):
#         return self.pad_char

#     @property
#     def id_GO(self):
#         return 1

#     @property
#     def id_EOS(self):
#         return 2
