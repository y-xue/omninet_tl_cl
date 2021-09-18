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
Authors: Subhojeet Pramanik, Priyanka Agrawal, Aman Hussain, Sayan Dutta

Dataloaders for standard datasets used in the paper

"""
import os
import torch
import pickle
import cv2
import numpy as np
import json
import tqdm
import random
from sklearn.model_selection import train_test_split
from PIL import Image
from bpemb import BPEmb
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import pandas as pd
# the following is required for the lib to work in terminal env
import matplotlib

matplotlib.use("agg", warn=False, force=True)
from .cocoapi.coco import COCO


from .vqa.vqa import VQA
import time
# from claims import Claim

from queue import Queue
"""
from train import args

print('dl.py args', args)


def seed_torch(seed=615):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

print('dl.py sets seed')
seed_torch(int(args.data_seed))
"""

# class VideoDataset(Dataset):
#     r"""A Dataset for a folder of videos. Expects the directory structure to be
#     directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
#     of all file names, along with an array of labels, with label being automatically
#     inferred from the respective folder names.
#         Args:
#             dataset (str): Name of dataset. Defaults to 'ucf101'.
#             split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
#             clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
#             preprocess (bool): Determines whether to preprocess dataset. Default is False.
#     """

#     def __init__(self, data_dir, output_dir, dataset='ucf101', split='train', clip_len=16, preprocess=False):
#         self.root_dir, self.output_dir = data_dir, output_dir
#         folder = os.path.join(self.output_dir, split)
#         self.clip_len = clip_len
#         self.split = split
#         self.resize_height = 300
#         self.resize_width = 300
#         self.crop_size = 224
#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         # The following three parameters are chosen as described in the paper section 4.1

#         if not self.check_integrity():
#             raise RuntimeError('Dataset not found or corrupted.' +
#                                ' You need to download it from official website.')

#         if (not self.check_preprocess()) or preprocess:
#             print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
#             import sys
#             sys.exit('processing video')
#             self.preprocess()

#         # Obtain all the filenames of files inside all the class folders
#         # Going through each class folder one at a time
#         self.fnames, labels = [], []
#         for label in sorted(os.listdir(folder)):
#             for fname in os.listdir(os.path.join(folder, label)):
#                 self.fnames.append(os.path.join(folder, label, fname))
#                 labels.append(label)

#         assert len(labels) == len(self.fnames)
#         print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

#         # Prepare a mapping between the label names (strings) and indices (ints)
#         self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
#         # Convert the list of label names into an array of label indices
#         self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

#         if not os.path.exists('conf/hmdblabels.txt'):
#             with open('conf/hmdblabels.txt', 'w') as f:
#                 for id, label in enumerate(sorted(self.label2index)):
#                     f.writelines(str(id) + ' ' + label + '\n')


#     def __len__(self):
#         return len(self.fnames)

#     def __getitem__(self, index):
#         buffer = self.load_frames(self.fnames[index])
#         buffer = self.crop(buffer, self.clip_len, self.crop_size)
#         labels = np.array(self.label_array[index])

#         if self.split == 'train':
#             # Perform data augmentation
#             buffer = self.randomflip(buffer)
#         buffer = self.normalize(buffer)
#         buffer = self.to_tensor(buffer)
#         return torch.from_numpy(buffer), torch.from_numpy(labels).unsqueeze(0)

#     def check_integrity(self):
#         if not os.path.exists(self.root_dir):
#             return False
#         else:
#             return True
    
#     def randomflip(self, buffer):
#         """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

#         if np.random.random() < 0.5:
#             for i, frame in enumerate(buffer):
#                 frame = cv2.flip(buffer[i], flipCode=1)
#                 buffer[i] = cv2.flip(frame, flipCode=1)

#         return buffer

    
#     def check_preprocess(self):
#         # TODO: Check image size in output_dir
#         if not os.path.exists(self.output_dir):
#             return False
#         elif not os.path.exists(os.path.join(self.output_dir, 'train')):
#             return False

#         for ii, video_class in enumerate(os.listdir(os.path.join(self.output_dir, 'train'))):
#             for video in os.listdir(os.path.join(self.output_dir, 'train', video_class)):
#                 video_name = os.path.join(os.path.join(self.output_dir, 'train', video_class, video),
#                                     sorted(os.listdir(os.path.join(self.output_dir, 'train', video_class, video)))[0])
#                 image = cv2.imread(video_name)
#                 if np.shape(image)[0] != self.resize_height or np.shape(image)[1] != self.resize_width:
#                     return False
#                 else:
#                     break

#             if ii == 10:
#                 break

#         return True

#     def preprocess(self):
#         if not os.path.exists(self.output_dir):
#             os.mkdir(self.output_dir)
#             os.mkdir(os.path.join(self.output_dir, 'train'))
#             os.mkdir(os.path.join(self.output_dir, 'test'))

#         # Split train/val/test sets
#         for file in tqdm.tqdm(os.listdir(self.root_dir)):
#             file_path = os.path.join(self.root_dir, file)
#             video_files = [name for name in os.listdir(file_path)]
#             split_file=os.path.join('conf/hmdb','%s_test_split1.txt'%file)
#             train_files=[]
#             test_files=[]
#             with open(split_file,'r') as f:
#                 lines=f.readlines()
#                 for l in lines:
#                     f_name,split=l.strip().split(' ')
#                     if split=='1' or split=='0':
#                         train_files.append(f_name)
#                     elif split=='2':
#                         test_files.append(f_name)
#             train_dir = os.path.join(self.output_dir, 'train', file)
#             test_dir = os.path.join(self.output_dir, 'test', file)
  
#             if not os.path.exists(train_dir):
#                 os.mkdir(train_dir)
#             if not os.path.exists(test_dir):
#                 os.mkdir(test_dir)

#             for video in train_files:
#                 self.process_video(video, file, train_dir)

#             for video in test_files:
#                 self.process_video(video, file, test_dir)

#         print('Preprocessing finished.')
        
        
#     def normalize(self, buffer):
#         buffer=buffer/255
#         for i, frame in enumerate(buffer):
#             frame -= np.array([[[0.485, 0.456, 0.406]]])
#             frame /= np.array([[[0.229, 0.224, 0.225]]])
#             buffer[i] = frame

#         return buffer
    
#     def to_tensor(self, buffer):
#         return buffer.transpose((0, 3, 1, 2))
    
#     def crop(self, buffer, clip_len, crop_size):
#         # randomly select time index for temporal jittering
#         if buffer.shape[0] - clip_len>0 and self.split=='train':
#             time_index = np.random.randint(buffer.shape[0] - clip_len)
#         else:
#             time_index=0
#         # Randomly select start indices in order to crop the video
#         if self.split=='train':
#             height_index = np.random.randint(buffer.shape[1] - crop_size)
#             width_index = np.random.randint(buffer.shape[2] - crop_size)
#         else:
#             height_index=0
#             width_index=0

#         # Crop and jitter the video using indexing. The spatial crop is performed on
#         # the entire array, so each frame is cropped in the same location. The temporal
#         # jitter takes place via the selection of consecutive frames
#         buffer = buffer[time_index:time_index + clip_len,
#                  height_index:height_index + crop_size,
#                  width_index:width_index + crop_size, :]

#         return buffer
    
#     def process_video(self, video, action_name, save_dir):
#         # Initialize a VideoCapture object to read video data into a numpy array
#         video_filename = video.split('.')[0]
#         if not os.path.exists(os.path.join(save_dir, video_filename)):
#             os.mkdir(os.path.join(save_dir, video_filename))

#         capture = cv2.VideoCapture(os.path.join(self.root_dir, action_name, video))

#         frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
#         frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
#         frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

#         # Make sure splited video has at least 16 frames
#         EXTRACT_FREQUENCY = 4
#         if frame_count // EXTRACT_FREQUENCY <= 16:
#             EXTRACT_FREQUENCY -= 1
#             if frame_count // EXTRACT_FREQUENCY <= 16:
#                 EXTRACT_FREQUENCY -= 1
#                 if frame_count // EXTRACT_FREQUENCY <= 16:
#                     EXTRACT_FREQUENCY -= 1

#         count = 0
#         i = 0
#         retaining = True

#         while (count < frame_count and retaining):
#             retaining, frame = capture.read()
#             if frame is None:
#                 continue

#             if count % EXTRACT_FREQUENCY == 0:
#                 if (frame_height != self.resize_height) or (frame_width != self.resize_width):
#                     frame = cv2.resize(frame, (self.resize_width, self.resize_height))
#                 cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
#                 i += 1
#             count += 1

#         # Release the VideoCapture once it is no longer needed
#         capture.release()


#     def load_frames(self, file_dir):
#         frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
#         frame_count = len(frames)
#         buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
#         for i, frame_name in enumerate(frames):
#             frame = np.array(cv2.imread(frame_name)).astype(np.float64)
#             buffer[i] = frame

#         return buffer


# def hmdb_batchgen(data_dir,process_dir,num_workers=0,batch_size=1,test_batch_size=1,clip_len=16):
#         dataset=VideoDataset(data_dir, process_dir, split='train',dataset='hmdb',clip_len=clip_len)
#         test_dataset=VideoDataset(data_dir, process_dir, split='test',dataset='hmdb',clip_len=clip_len)
#         dataloader=DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True,
#                    )
#         itr = iter(cycle(dataloader))
#         test_dl= DataLoader(test_dataset, num_workers=num_workers, batch_size=test_batch_size,
#                                       drop_last=False)
#         return itr,test_dl

class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.
        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    """

    def __init__(self, data_dir, output_dir, video_fns_labels=None, all_class=False, dataset='ucf101', split='train', clip_len=16, preprocess=False):
        self.root_dir, self.output_dir = data_dir, output_dir
        # folder = os.path.join(self.output_dir, split)
        self.clip_len = clip_len
        self.split = split
        self.resize_height = 300
        self.resize_width = 300
        self.crop_size = 224
        self.video_fns_labels = video_fns_labels
        self.all_class = all_class
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # The following three parameters are chosen as described in the paper section 4.1

        if all_class and video_fns_labels is not None:
            labels = []
            for label in sorted(os.listdir(os.path.join(output_dir, 'train'))):
                labels.append(label)

            self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
            # print(self.label2index)
            self.fnames = self.video_fns_labels[self.split]['fns']
            # if split == 'train':
            #     self.fnames += self.video_fns_labels['val']['fns']
            
            print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        if video_fns_labels is None:
            if not self.check_integrity():
                raise RuntimeError('Dataset not found or corrupted.' +
                                   ' You need to download it from official website.')
            if preprocess:
            # if (not self.check_preprocess()) or preprocess:
                print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
                self.preprocess()

            # Obtain all the filenames of files inside all the class folders
            # Going through each class folder one at a time
            self.fnames, labels = [], []
            for label in sorted(os.listdir(folder)):
                for fname in os.listdir(os.path.join(folder, label)):
                    self.fnames.append(os.path.join(folder, label, fname))
                    labels.append(label)

            assert len(labels) == len(self.fnames)
            print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

            # Prepare a mapping between the label names (strings) and indices (ints)
            self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
            # Convert the list of label names into an array of label indices
            self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

            if not os.path.exists('conf/hmdblabels.txt'):
                with open('conf/hmdblabels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id) + ' ' + label + '\n')


    def __len__(self):
        return len(self.fnames)
        # if self.video_fns_labels is None:
        #     return len(self.fnames)
        # else:
        #     return len(self.video_fns_labels[self.split]['labels'])

    def __getitem__(self, index):
        if self.video_fns_labels is None:
            buffer = self.load_frames(self.fnames[index])
            labels = np.array(self.label_array[index])
        else:
            if self.split == 'train' or self.split == 'val':
                process_split = 'train'
            else:
                process_split = 'test'

            # print(self.video_fns_labels[self.split]['fns'][index].split('.')[0])
            # print(self.split)

            buffer = self.load_frames(os.path.join(self.output_dir, process_split, 
                            self.fnames[index].split('.')[0]))

            if self.all_class:
                labels = np.array(self.label2index[self.fnames[index].split('/')[0]])
            else:
                labels = np.array(self.video_fns_labels[self.split]['labels'][index])

            # print('video_fn: ', self.video_fns_labels[self.split]['fns'][index].split('.')[0])
            # print('label: ', labels)

        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        
        if self.split == 'train':
            # Perform data augmentation
            buffer = self.randomflip(buffer)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        # print(buffer.shape)
        return torch.from_numpy(buffer), torch.from_numpy(labels).unsqueeze(0)

    def check_integrity(self):
        if not os.path.exists(self.root_dir):
            return False
        else:
            return True
    
    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    
    def check_preprocess(self):
        # TODO: Check image size in output_dir
        if not os.path.exists(self.output_dir):
            return False
        elif not os.path.exists(os.path.join(self.output_dir, 'train')):
            return False

        for ii, video_class in enumerate(os.listdir(os.path.join(self.output_dir, 'train'))):
            for video in os.listdir(os.path.join(self.output_dir, 'train', video_class)):
                video_name = os.path.join(os.path.join(self.output_dir, 'train', video_class, video),
                                    sorted(os.listdir(os.path.join(self.output_dir, 'train', video_class, video)))[0])
                image = cv2.imread(video_name)
                if np.shape(image)[0] != self.resize_height or np.shape(image)[1] != self.resize_width:
                    return False
                else:
                    break

            if ii == 10:
                break

        return True

    def preprocess(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            os.mkdir(os.path.join(self.output_dir, 'train'))
            os.mkdir(os.path.join(self.output_dir, 'val'))
            os.mkdir(os.path.join(self.output_dir, 'test'))

        # Split train/val/test sets
        for file in tqdm.tqdm(os.listdir(self.root_dir)):
            file_path = os.path.join(self.root_dir, file)
            video_files = [name for name in os.listdir(file_path)]
            split_file=os.path.join('conf/hmdb','%s_test_split1.txt'%file)
            train_files=[]
            test_files=[]
            val_files=[]
            with open(split_file,'r') as f:
                lines=f.readlines()
                for l in lines:
                    f_name,split=l.strip().split(' ')
                    if split=='1':
                        train_files.append(f_name)
                    elif split=='0':
                        val_files.append(f_name)
                    elif split=='2':
                        test_files.append(f_name)
            train_dir = os.path.join(self.output_dir, 'train', file)
            val_dir = os.path.join(self.output_dir, 'val', file)
            test_dir = os.path.join(self.output_dir, 'test', file)
  
            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(val_dir):
                os.mkdir(val_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)

            for video in train_files:
                self.process_video(video, file, train_dir)

            for video in val_files:
                self.process_video(video, file, val_dir)

            for video in test_files:
                self.process_video(video, file, test_dir)

        print('Preprocessing finished.')
        
        
    def normalize(self, buffer):
        buffer=buffer/255
        for i, frame in enumerate(buffer):
            frame -= np.array([[[0.485, 0.456, 0.406]]])
            frame /= np.array([[[0.229, 0.224, 0.225]]])
            buffer[i] = frame

        return buffer
    
    def to_tensor(self, buffer):
        return buffer.transpose((0, 3, 1, 2))
    
    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        if buffer.shape[0] - clip_len>0 and self.split=='train':
            time_index = np.random.randint(buffer.shape[0] - clip_len)
        else:
            time_index=0
        # Randomly select start indices in order to crop the video
        if self.split=='train':
            height_index = np.random.randint(buffer.shape[1] - crop_size)
            width_index = np.random.randint(buffer.shape[2] - crop_size)
        else:
            height_index=0
            width_index=0

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer
    
    def process_video(self, video, action_name, save_dir):
        # Initialize a VideoCapture object to read video data into a numpy array
        video_filename = video.split('.')[0]
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))

        capture = cv2.VideoCapture(os.path.join(self.root_dir, action_name, video))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Make sure splited video has at least 16 frames
        EXTRACT_FREQUENCY = 4
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= 16:
                    EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        retaining = True

        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % EXTRACT_FREQUENCY == 0:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
                i += 1
            count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()


    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame

        return buffer

def hmdb_batchgen(data_dir,process_dir,num_workers=1,batch_size=1,test_batch_size=1,clip_len=16,all_class=True):
    with open(os.path.join(data_dir, 'hmdb_binary'), 'rb') as fp:
        video_fns_labels = pickle.load(fp)

    dataset=VideoDataset(data_dir, process_dir, video_fns_labels=video_fns_labels, all_class=all_class, split='train',dataset='hmdb',clip_len=clip_len)
    val_dataset=VideoDataset(data_dir, process_dir, video_fns_labels=video_fns_labels, all_class=all_class, split='val',dataset='hmdb',clip_len=clip_len)
    # test_dataset=VideoDataset(data_dir, process_dir, video_fns_labels=video_fns_labels, all_class=all_class, split='test',dataset='hmdb',clip_len=clip_len)
    tr_dl=DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, 
        )
    val_dl= DataLoader(val_dataset, num_workers=num_workers, batch_size=test_batch_size,
                                  drop_last=False)
    itr = iter(cycle(tr_dl))
    # test_dl= DataLoader(test_dataset, num_workers=num_workers, batch_size=test_batch_size,
    #                               drop_last=False)
    return itr, val_dl #, test_dl

class coco_cap_dataset(Dataset):
    def __init__(self, ann_file, image_dir,transforms=None,max_words=40):
        caption = COCO(ann_file)
        self.inputs = []
        self.outputs = []
        ann_ids = caption.getAnnIds()
        for idx, a in tqdm.tqdm(enumerate(ann_ids),'Loading MSCOCO to memory'):
            c = caption.loadAnns(a)[0]
            words = c['caption']
            if len(words.split(' '))<=max_words:
                img_file = os.path.join(image_dir, '%012d.jpg' % (c['image_id']))
                self.inputs.append(img_file)
                self.outputs.append(words)
        self.transform = transforms
        self.N=len(self.outputs)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        img = Image.open(self.inputs[idx])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img).convert("RGB")
        cap = self.outputs[idx]
        # returns the dictionary of images and captions
        return {'img': img, 'cap': cap}


def coco_cap_batchgen(caption_dir, image_dir,num_workers=0, batch_size=1):
        # transformations for the images
        train_ann=os.path.join(caption_dir,'captions_train2017.json')
        val_ann=os.path.join(caption_dir,'captions_val2017.json')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transformer = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.4,.4,.4),
            transforms.ToTensor(),
            normalize,
        ])
        # the dataset
        dataset = coco_cap_dataset(train_ann, image_dir, transforms=transformer)
        # the data loader
        dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True,
                                     collate_fn=coco_collate_fn, drop_last=True,pin_memory=True)
       
        # the iterator over data loader
        itr = iter(cycle(dataloader))
        
        val_tfms = transforms.Compose([
                                        transforms.Resize(int(224*1.14)),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize,
                                    ])
        val_dataset = coco_cap_dataset( val_ann, image_dir, transforms=val_tfms,max_words=5000)
        val_dl= DataLoader(val_dataset, num_workers=num_workers, batch_size=int(batch_size/2),
                                     collate_fn=coco_collate_fn, drop_last=False,pin_memory=True)
        return itr, val_dl
                    

    
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def coco_collate_fn(data):
    # collate function for the data loader
    collate_images = []
    collate_cap = []
    max_len = 0
    for d in data:
        collate_images.append(d['img'])
        collate_cap.append(d['cap'])
    collate_images = torch.stack(collate_images, dim=0)
    # return a dictionary of images and captions
    return {
        'img': collate_images,
        'cap': collate_cap
    }

# def ans_in_vocab(answer, ans_to_id):
#     for ans in answer[0]['answers']:
#         if ans['answer'] in ans_to_id:
#             return ans['answer']
#     return None

def round_box_coordinates(box, height, width):
    bbox = []
    for i, bbox_coord in enumerate(box):
        modified_bbox_coord = float(bbox_coord)
        if i % 2:
            modified_bbox_coord *= height
        else:
            modified_bbox_coord *= width
        modified_bbox_coord = int(modified_bbox_coord)
        bbox.append(modified_bbox_coord)
    return bbox

class vqa_dataset(Dataset):
    def __init__(self, ques_file, ann_file, image_dir,vocab_file, structured=None, transforms=None, ids=None, boxes_dir=None, unstructured_as_structured=False):
        vqa = VQA(annotation_file=ann_file, question_file=ques_file)
        self.imgs = []
        self.ques = []
        self.ans = []
        self.structured = []
        self.boxes = []

        boxes_dict = {}

        # if boxes_fn is not None:
        #     with open(boxes_fn, 'r') as f:
        #         boxes_json = json.load(f)
            
        #     boxes_dict = {}

        #     for bbox in boxes_json:
        #         img_id = bbox['image_id'] 
        #         if img_id in boxes_dict:
        #             boxes_dict[bbox['image_id']].append(bbox['bbox_normalized'])
        #         else:
        #             boxes_dict[bbox['image_id']] = [bbox['bbox_normalized']]

        with open(vocab_file,'rb') as f:
            ans_to_id,id_to_ans=pickle.loads(f.read())
        # load the questions
        ques = vqa.questions['questions']
        # for tracking progress
        # for every question
        for i,x in tqdm.tqdm(enumerate(ques),'Loading VQA data to memory'):
            if ids is not None and i not in ids:
                continue
            # get the path
            answer = vqa.loadQA(x['question_id'])
            m_a=answer[0]['multiple_choice_answer']
            # a = ans_in_vocab(answer, ans_to_id)
            if m_a in ans_to_id: # or a is not None:
                img_file = os.path.join(image_dir, '%012d.jpg' % (x['image_id']))
                self.imgs.append(img_file)
                # get the vector representation
                words = x['question']
                self.ques.append(words)
                self.ans.append(ans_to_id[m_a])

                if boxes_dir is not None:
                    if x['image_id'] in boxes_dict:
                        boxes_json = boxes_dict[x['image_id']]
                    else:
                        with open(os.path.join(boxes_dir, '%s.json'%(x['image_id'])), 'r') as f:
                            boxes_json = json.load(f)

                        boxes_dict[x['image_id']] = boxes_json

                    if len(boxes_json) == 0:
                        self.boxes.append(None)
                    else:
                        bboxes = []
                        for bbox in boxes_json:
                            bboxes.append(bbox['bbox_normalized'])
                        self.boxes.append(bboxes)

                if structured is not None:
                    if unstructured_as_structured:
                        self.structured.append(structured[i])
                    else:
                        self.structured.append(structured[x['question_id']])
                        # self.structured.append(structured[x['image_id']])
                    
        self.transform = transforms
        self.N=len(self.ques)
        self.resuming = False
    
    def resume_on(self):
        # self.tmp_img = Image.open(self.imgs[0])
        # self.tmp_img = self.tmp_img.convert('RGB')
        self.resuming = True

    def resume_off(self):
        self.resuming = False

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        if self.resuming:
            # if self.transform is not None:
            #     img = self.transform(self.tmp_img)
            # else:
            #     img = transforms.ToTensor()(self.tmp_img).convert("RGB")

            return {'img': torch.tensor([0]), 'ques': None, 'ans': torch.tensor([1]), 'struct': None, 'bbox_mask': None}

        # start_time = time.time()
        img = Image.open(self.imgs[idx])
        # end_time = time.time()
        # print('reading img takes {:.2f}s'.format(end_time - start_time))

        # start_time = time.time()
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img).convert("RGB")

        bbox_mask = None
        if len(self.boxes) != 0:
            if self.boxes[idx] is None:
                # print('no bounding_boxes found')
                bbox_mask = torch.zeros((7,7))
            else:
                bbox_mask = torch.ones((7,7))
                for x1, y1, x2, y2 in [round_box_coordinates(box, 7, 7) for box in self.boxes[idx]]:
                    bbox_mask[x1:(x2+1),y1:(y2+1)] = 0
                # bbox_mask[bbox_mask!=-1] = 1.
                # bbox_mask[bbox_mask==-1] = 0.
            
            # print(bbox_mask)
            bbox_mask = bbox_mask.reshape(-1).type(torch.ByteTensor)
            # print(bbox_mask.shape)

        # end_time = time.time()
        # print('transoforming img takes {:.2f}s'.format(end_time - start_time))

        ques = self.ques[idx]
        ans = self.ans[idx]
        struct = None
        if len(self.structured) != 0:
            struct = torch.as_tensor(self.structured[idx]).float()
            # struct = torch.tensor(self.structured[idx])
        # finally return dictionary of images, questions and answers
        # for a given index
        return {'img': img, 'ques': ques, 'ans': ans, 'struct': struct, 'bbox_mask': bbox_mask}


def vqa_batchgen(vqa_dir, image_dir, num_workers=0, batch_size=1, structured_path=None, with_val=False, use_boxes=False, make_tr_dl_iter=True, drop_last_tr=True, unstructured_as_structured=False, data_seed=1029):
    random.seed(data_seed)
    np.random.seed(data_seed)
    torch.manual_seed(data_seed)

    # a transformation for the images
    vqa_train_ques=os.path.join(vqa_dir,'v2_OpenEnded_mscoco_train2014_questions.json')
    vqa_train_ann=os.path.join(vqa_dir,'v2_mscoco_train2014_annotations.json')
    vqa_val_ques=os.path.join(vqa_dir,'v2_OpenEnded_mscoco_val2014_questions.json')
    vqa_val_ann=os.path.join(vqa_dir,'v2_mscoco_val2014_annotations.json')

    if use_boxes:
        boxes_dir = os.path.join('/files/yxue/research/allstate/data/coco','coco_boxes/boxes')          
    else:
        boxes_dir = None

    train_idx = val_idx = None
    if with_val:
        with open(os.path.join(vqa_dir, 'train_idx.pkl'), 'rb') as f:
            train_idx = pickle.load(f)
        with open(os.path.join(vqa_dir, 'val_idx.pkl'), 'rb') as f:
            val_idx = pickle.load(f)
        

    tr_structured = None
    val_structured = None
    if structured_path is not None:

        if unstructured_as_structured:
            with open(os.path.join(structured_path, 'synthetic_train_structured_data.dict'), 'rb') as f:
                tr_structured = pickle.load(f)
            with open(os.path.join(structured_path, 'synthetic_val_structured_data.dict'), 'rb') as f:
                val_structured = pickle.load(f)
        else:
            with open(os.path.join(structured_path, 'synthetic_structured_data_normed.dict'), 'rb') as f:
                synthetic_structured_data = pickle.load(f)
            tr_structured = synthetic_structured_data['train']
            val_structured = synthetic_structured_data['val']

    vocab_file=os.path.join('conf/vqa_vocab.pkl')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformer = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(.4, .4, .4),
        transforms.ToTensor(),
        normalize,
    ])
    # the dataset
    dataset = vqa_dataset(vqa_train_ques, vqa_train_ann, image_dir, vocab_file, structured=tr_structured, transforms=transformer, ids=train_idx, boxes_dir=boxes_dir, unstructured_as_structured=unstructured_as_structured)
    # the data loader
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True,
                                 collate_fn= vqa_collate_fn, drop_last=drop_last_tr,pin_memory=True)
    print('vqa # of training batches:', len(dataloader))
    if make_tr_dl_iter:
        # the iterator
        itr = iter(cycle(dataloader))
    else:
        itr = dataloader

    val_tfms = transforms.Compose([
        transforms.Resize(int(224 * 1.14)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = vqa_dataset(vqa_val_ques, vqa_val_ann, image_dir, vocab_file, structured=val_structured, transforms=val_tfms, boxes_dir=boxes_dir, unstructured_as_structured=unstructured_as_structured)
    # the data loader
    test_dataloader = DataLoader(test_dataset, num_workers=num_workers, batch_size=int(batch_size/2), shuffle=True,
                                 collate_fn=vqa_collate_fn, drop_last=False)
    
    if with_val:
        # the dataset
        val_dataset = vqa_dataset(vqa_train_ques, vqa_train_ann, image_dir, vocab_file, structured=tr_structured, transforms=transformer, ids=val_idx, boxes_dir=boxes_dir, unstructured_as_structured=unstructured_as_structured)
        # the data loader
        val_dataloader = DataLoader(val_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True,
                                     collate_fn= vqa_collate_fn, drop_last=False,pin_memory=True)
        return itr,val_dataloader,test_dataloader

    return itr,test_dataloader,None
        


def vqa_collate_fn(data):
    # the collate function for dataloader
    collate_images = []
    collate_ques = []
    collate_ans=[]
    collate_struct = []
    collate_bbox_mask = []
    for d in data:
        collate_images.append(d['img'])
        collate_ques.append(d['ques'])
        collate_ans.append((d['ans']))
        if d['struct'] is not None:
            collate_struct.append((d['struct']))
        if d['bbox_mask'] is not None:
            collate_bbox_mask.append(d['bbox_mask'])
    collate_images = torch.stack(collate_images, dim=0)
    collate_ans=torch.tensor(collate_ans).reshape([-1,1])

    if len(collate_struct) != 0:
        collate_struct = torch.stack(collate_struct, dim=0)
    if len(collate_bbox_mask) != 0:
        collate_bbox_mask = torch.stack(collate_bbox_mask, dim=0)
    else:
        collate_bbox_mask = None
    # return a dictionary of images and captions
    # return dict of collated images answers and questions
    return {
        'img': collate_images,
        'ques': collate_ques,
        'ans': collate_ans,
        'struct': collate_struct,
        'bbox_mask': collate_bbox_mask
    }

class Claim:
    def __init__(self, seq, weights, sample_idx=None, scale_weights=False):
        self.seq = seq
        self.weights = weights
        self.resize_height = 300
        self.resize_width = 300
        self.crop_size = 224
        self.clip_len = 16
        self.scale_weights = scale_weights

        self.sample_idx_by_instance = None
        if sample_idx is not None:
            # print('normalizing sample weights by instance...')
            sample_idx_by_instance={}
            sample_idx_by_instance['train']={}
            sample_idx_by_instance['val']={}
            sample_idx_by_instance['test']={}

            for mode in ['train','val','test']:
                for c in [0,1]:
                    sample_idx_by_instance[mode][c] = self.group_sample_idx_by_instance(sample_idx, mode=mode, c=c)
            
            self.assign_weights(sample_idx_by_instance)
            self.self_normalize(sample_idx_by_instance)

            self.sample_idx_by_instance = sample_idx_by_instance
            
            # for idx in [0, 10, 100, 1000]:
            #     sample = sample_idx['train']['0-1-0'][idx]
            #     print(sample)
            #     print(self.sample_idx_by_instance['train'][sample[2]][sample[0]])

    # def get_image_files(self, image_dir):
    #     return 0

    # def get_images(self, image_dir, h=32, w=32):
    #     # imgs = np.loadtxt(image_dir+'/cats_and_dogs',delimiter=',')
    #     with open(image_dir+'/cats_and_dogs', 'rb') as f:
    #         imgs = pickle.load(f)
    #     train_imgs = imgs['train']['data'].reshape(-1,3,h,w).transpose((0,2,3,1))
    #     test_imgs = imgs['test']['data'].reshape(-1,3,h,w).transpose((0,2,3,1))
    #     return train_imgs, test_imgs

    # def read_txt(self, fn):
    #     with open(fn, 'r+', encoding="utf-8") as f:
    #         return f.read()

    # def list_files(self, d):
    #     return [f for f in os.listdir(d) if os.path.isfile(os.path.join(d,f))]

    # def get_text(self, text_dir):
    #     pos_files = self.list_files(text_dir+'/train/pos')
    #     neg_files = self.list_files(text_dir+'/train/neg')

    #     text = [self.read_txt(os.path.join(text_dir,'train/pos', fn)) for fn in pos_files]
    #     text.extend([self.read_txt(os.path.join(text_dir,'train/neg', fn)) for fn in neg_files])
    #     return text
    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    def normalize(self, buffer):
        buffer=buffer/255
        for i, frame in enumerate(buffer):
            frame -= np.array([[[0.485, 0.456, 0.406]]])
            frame /= np.array([[[0.229, 0.224, 0.225]]])
            buffer[i] = frame

        return buffer
    
    def to_tensor(self, buffer):
        return buffer.transpose((0, 3, 1, 2))
    
    def crop(self, buffer, clip_len, crop_size, split):
        # randomly select time index for temporal jittering
        if buffer.shape[0] - clip_len>0 and split=='train':
            time_index = np.random.randint(buffer.shape[0] - clip_len)
        else:
            time_index=0
        # Randomly select start indices in order to crop the video
        if split=='train':
            height_index = np.random.randint(buffer.shape[1] - crop_size)
            width_index = np.random.randint(buffer.shape[2] - crop_size)
        else:
            height_index=0
            width_index=0

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame

        return buffer

    def load_video(self, fn, split):
        video = self.load_frames(fn)
        video = self.crop(video, self.clip_len, self.crop_size, split)
        if split == 'train':
            video = self.randomflip(video)
        video = self.normalize(video)
        video = self.to_tensor(video)
        return torch.from_numpy(video)

    def get_videos(self, video_dir, video_process_dir):
        # video_dir = '/Users/ye/Documents/research/allstate/data/hmdb2'
        with open(os.path.join(video_dir, 'video_fns_labels'), 'rb') as fp:
            video_fns_labels = pickle.load(fp)

        train_videos = [self.load_video(os.path.join(video_process_dir,'train',f.split('.')[0]), 'train') for f in video_fns_labels['train']['fns']]#[self.load_frames(os.path.join(video_process_dir,'train',f.split('.')[0])) for f in video_fns_labels['train']['fns']]
        val_videos = [self.load_video(os.path.join(video_process_dir,'val',f.split('.')[0]), 'val') for f in video_fns_labels['val']['fns']]#[self.load_frames(os.path.join(video_process_dir,'train',f.split('.')[0])) for f in video_fns_labels['train']['fns']]#[self.load_frames(os.path.join(video_process_dir,'val',f.split('.')[0])) for f in video_fns_labels['val']['fns']]
        test_videos = [self.load_video(os.path.join(video_process_dir,'test',f.split('.')[0]), 'test') for f in video_fns_labels['test']['fns']]#[self.load_frames(os.path.join(video_process_dir,'train',f.split('.')[0])) for f in video_fns_labels['train']['fns']]#[self.load_frames(os.path.join(video_process_dir,'test',f.split('.')[0])) for f in video_fns_labels['test']['fns']]

        return train_videos, val_videos, test_videos
        # return {
        #     'train': {'data': train_videos, 'labels': video_fns_labels['train']['labels']},
        #     'val': {'data': val_videos, 'labels': video_fns_labels['val']['labels']},
        #     'test': {'data': test_videos, 'labels': video_fns_labels['test']['labels']},
        # }

    def group_sample_idx_by_instance(self, sample_idx, mode='train', c=0):
        sample_idx = sample_idx[mode]
        sample_idx_by_instance = {}
        for seq in sample_idx.keys():
            for sample in sample_idx[seq]:
                cid = sample[0]
                if sample[2] != c:
                    continue
                if cid in sample_idx_by_instance:
                    sample_idx_by_instance[cid].append(sample[1:])
                else:
                    sample_idx_by_instance[cid] = [sample[1:]]
        return sample_idx_by_instance

    def assign_weights(self, sample_idx_by_instance):
        """
        weights: {nModes: w}
        """
        for mode in sample_idx_by_instance:
            for c in sample_idx_by_instance[mode]:
                for instance_id in sample_idx_by_instance[mode][c]:
                    for sid,sample in enumerate(sample_idx_by_instance[mode][c][instance_id]):
                        sample.append(self.get_weight(sample))
                        sample_idx_by_instance[mode][c][instance_id][sid] = sample
        
    def self_normalize(self, sample_idx_by_instance):
        for mode in sample_idx_by_instance:
            for c in sample_idx_by_instance[mode]:
                for instance_id in sample_idx_by_instance[mode][c]:
                    weights = [sample[-1] for sample in sample_idx_by_instance[mode][c][instance_id]]
                    normalized_weights = np.array(weights)/sum(weights)
                    if self.scale_weights:
                        normalized_weights = normalized_weights * len(self.seq)
                    
                    for sid,sample in enumerate(sample_idx_by_instance[mode][c][instance_id]):
                        sample[-1] = normalized_weights[sid]
                        sample_idx_by_instance[mode][c][instance_id][sid] = sample

    def get_mode_idx(self, sample, mode):
        sample = sample[-len(self.seq):]
        ids = []
        for i in np.where(np.array(list(self.seq))==mode)[0]:
            if sample[i] != -1:
                ids.append(sample[i])
        return ids

    def get_label(self, sample):
        # sample: [instance_id,t,c,...]
        return sample[2]

    def get_weight(self, sample, norm=False, mode='train'):
        if norm:
            return self.get_norm_weights(mode, sample)

        # sample: [instance_id,t,c,...]
        n = len(self.seq)
        mode_cnt = int(n - sum(np.array(sample[-n:])==-1))

        if self.scale_weights:
            return len(self.seq) * self.weights[str(mode_cnt)]
        return self.weights[str(mode_cnt)]

    def get_norm_weights(self, mode, sample):
        instance_id = sample[0]
        t = sample[1]
        c = sample[2]
        for wsample in self.sample_idx_by_instance[mode][c][instance_id]:
            if wsample[0] == t:
                return wsample[-1]
        return -1

class mm_dataset(Dataset):
    def __init__(self, claim, full_seq, seq, imgs, text, videos, sample_idx, 
        video_process_dir=None, video_fns_labels=None, 
        mode='train', transforms=None, norm_weights=False, 
        pick_sample_by_intsance_id=False, stop_overfitted_ins=False, 
        shuffle=True, split='train', seq_count=True):
        # sample_idx:
        #  {seq_idx: [instance_id, t, c, I, T, T, I, ...]}

        # print('mm_dataset %s'%seq)

        self.claim = claim
        self.imgs = imgs #self.claim.get_images(image_dir)
        self.text = text #self.claim.get_text(text_dir)
        self.videos = videos
        self.video_process_dir = video_process_dir
        self.video_fns_labels = video_fns_labels
        self.split = split
        self.sample_idx = sample_idx[seq]
        self.transform = transforms
        self.N=len(self.sample_idx)
        # print(self.N)
        if seq_count:
            self.nimgs = int(seq.split('-')[0])#seq.count('I')
            self.ntext = int(seq.split('-')[1])#seq.count('T')
            self.nvideo = int(seq.split('-')[2])
        else:
            self.nimgs = seq.count('I')
            self.ntext = seq.count('T')
            self.nvideo = seq.count('V')
        self.seq = seq
        self.norm_weights = norm_weights
        self.mode = mode
        self.pick_sample_by_intsance_id = pick_sample_by_intsance_id
        self.stop_overfitted_ins = stop_overfitted_ins

        self.marked = set([])
        self.ins_ids = set([])

        self.sample_idx_queue = Queue(maxsize = self.N)
        self.qsize = self.N

        perm = np.arange(self.N)

        if shuffle:
            np.random.shuffle(perm)

        for i in perm:
            self.sample_idx_queue.put(i)
            ins_id = '%s_%s'%(self.sample_idx[i][0], self.sample_idx[i][2])
            if ins_id not in self.ins_ids:
                self.ins_ids.add(ins_id)

        self.n_popped = 0
        self.selected_ins_ids = []
    
    # def size(self):
    #     return self.qsize
        
    def get_marked(self):
        return self.marked

    def set_marked(self, marked):
        self.marked = marked

    def remove_marked_ins(self):
        n_removed = 0
        for i in range(self.qsize):
            cur_idx = self.sample_idx_queue.get()
            ins_id = '%s_%s'%(self.sample_idx[cur_idx][0], self.sample_idx[cur_idx][2])
            if ins_id not in self.marked:
                self.sample_idx_queue.put(cur_idx)
            else:
                n_removed += 1

        self.qsize -= n_removed

    def get_ins_ids(self):
        return self.ins_ids

    def get_selected_ins_ids(self):
        return self.selected_ins_ids

    def empty(self):
        if self.sample_idx_queue.empty() and self.qsize != 0:
            print(self.qsize)
            raise NameError('qsize is not 0 but queue is empty')

        return self.sample_idx_queue.empty()

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        cur_idx = idx

        if self.pick_sample_by_intsance_id:
            while True:
                cur_idx = self.sample_idx_queue.get()
                # print(cur_idx)
                self.sample_idx_queue.put(cur_idx)
                self.n_popped += 1
                # self.n_popped = (self.n_popped + 1) % self.N
                
                ins_id = '%s_%s'%(self.sample_idx[cur_idx][0], self.sample_idx[cur_idx][2])
                if ins_id not in self.marked:
                    self.marked.add(ins_id)
                    if self.n_popped == self.N:
                        self.n_popped = 0
                    self.selected_ins_ids.append(ins_id)
                    break

                if self.n_popped == self.N:
                    self.n_popped = 0
                    self.marked = set([])

            idx = cur_idx

        if self.stop_overfitted_ins:
            idx = self.sample_idx_queue.get()
            self.sample_idx_queue.put(idx)

            # while not self.sample_idx_queue.empty():
            #     cur_idx = self.sample_idx_queue.get()

            #     ins_id = '%s_%s'%(self.sample_idx[cur_idx][0], self.sample_idx[cur_idx][2])
                
            #     if ins_id not in self.marked:
            #         self.sample_idx_queue.put(cur_idx)
            #         break

            # idx = cur_idx

        img_set = None
        # print(self.seq, idx)
        if self.nimgs != 0:
            image_ids = self.claim.get_mode_idx(self.sample_idx[idx], 'I')
            imgs = [self.transforming(Image.fromarray(self.imgs[i].astype(np.uint8))) for i in image_ids]
            img_set = np.stack(imgs) #np.empty((self.nimgs, c, h, w), np.dtype('float32'))
            img_set = torch.from_numpy(img_set)
            # print('from_numpy')

        text_set = None
        if self.ntext != 0:
            text_ids = self.claim.get_mode_idx(self.sample_idx[idx], 'T')
            text_set = [self.text[i] for i in text_ids] 
            # text_set = [self.truncate(self.text[i]) for i in text_ids]
            # text_set = [self.removeNonAscii(self.text[i]) for i in text_ids]
            # text_set = [self.truncate(self.removeNonAscii(self.text[i]), max_len=300) for i in text_ids]

        video_set = None
        if self.nvideo != 0:
            video_ids = self.claim.get_mode_idx(self.sample_idx[idx], 'V')

            if self.videos is not None:
                video_set = torch.from_numpy(np.stack([self.videos[i] for i in video_ids]))
            else:
                if self.split == 'train' or self.split == 'val':
                    process_split = 'train'
                else:
                    process_split = 'test'

                video_set = torch.from_numpy(
                    np.stack([self.claim.load_video(
                        os.path.join(self.video_process_dir, process_split, 
                            self.video_fns_labels[self.split]['fns'][i].split('.')[0]), self.split) 
                    for i in video_ids]))

        # finally return dictionary of images, questions and answers
        # for a given index
        return {
            'img': img_set, 
            'text': text_set, 
            'video': video_set,
            'label': self.claim.get_label(self.sample_idx[idx]), 
            'sample_weight': self.claim.get_weight(self.sample_idx[idx], norm=self.norm_weights, mode=self.mode)
        }

    def transforming(self, img):
        # img = Image.open(img)
        # img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img).convert("RGB")
        return img

    def truncate(self, sent, max_len=300):
        if len(sent) > max_len:
            return sent[:max_len]
        return sent

    def removeNonAscii(self, s):
        return "".join(i for i in s if ord(i)<128)

def mm_batchgen(full_seq, seq_lst, image_dir, text_dir, video_dir, video_process_dir, 
    predefined_sample_weights, sample_idx, num_workers=0, batch_size=5, 
    norm_weights=False, scale_weights=False, 
    pick_sample_by_intsance_id=False, stop_overfitted_ins=False,
    seq_count=True, drop_last_tr=True, reset_seed=None, data_seed=1029):
        if reset_seed is not None:
            data_seed += reset_seed
    
        random.seed(data_seed)
        np.random.seed(data_seed)
        torch.manual_seed(data_seed)

        # a transformation for the images
        print('mm_batchgen...')

        if norm_weights:
            claim = Claim(full_seq, predefined_sample_weights, sample_idx=sample_idx, scale_weights=scale_weights)
        else:
            claim = Claim(full_seq, predefined_sample_weights, sample_idx=None, scale_weights=scale_weights)

        print('Loading images...')
        h = w = 32
        start_time = time.time()
        # with open(os.path.join(image_dir, 'cats_and_dogs_TrValTe'), 'rb') as f:
        #     image_data = pickle.load(f)
        with open(os.path.join(image_dir, 'cifar10_binary_dataset_TrValTe'), 'rb') as f:
            image_data = pickle.load(f)
        end_time = time.time()
        print('Loading images takes: {:.2f}s\n'.format(end_time - start_time))
        
        train_imgs = image_data['train']['data'].reshape(-1,3,h,w).transpose((0,2,3,1))
        val_imgs = image_data['val']['data'].reshape(-1,3,h,w).transpose((0,2,3,1))
        test_imgs = image_data['test']['data'].reshape(-1,3,h,w).transpose((0,2,3,1))
        
        print('Loading text...')
        start_time = time.time()
        # text = claim.get_text(text_dir)
        with open(os.path.join(text_dir, 'train_text.dict'), 'rb') as f:
            train_text = pickle.load(f)['data']
        with open(os.path.join(text_dir, 'val_text.dict'), 'rb') as f:
            val_text = pickle.load(f)['data']
        with open(os.path.join(text_dir, 'test_text.dict'), 'rb') as f:
            test_text = pickle.load(f)['data']
        end_time = time.time()
        print('Loading text takes: {:.2f}s\n'.format(end_time - start_time))
        # print(len(train_text), len(test_text))

        # print('Loading videos...')
        # start_time = time.time()

        with open(os.path.join(video_dir, 'hmdb_binary1666'), 'rb') as fp:
            video_fns_labels = pickle.load(fp)

        # # train_videos, val_videos, test_videos = claim.get_videos(video_dir, video_process_dir)
        # # with open(os.path.join(video_process_dir, 'train_videos.lst'), 'wb') as f:
        # #     pickle.dump(train_videos, f)
        # # with open(os.path.join(video_process_dir, 'val_videos.lst'), 'wb') as f:
        # #     pickle.dump(val_videos, f)
        # # with open(os.path.join(video_process_dir, 'test_videos.lst'), 'wb') as f:
        # #     pickle.dump(test_videos, f)
        # with open(os.path.join(video_process_dir, 'train_videos.lst'), 'rb') as f:
        #     train_videos = pickle.load(f)
        # with open(os.path.join(video_process_dir, 'val_videos.lst'), 'rb') as f:
        #     val_videos = pickle.load(f)
        # with open(os.path.join(video_process_dir, 'test_videos.lst'), 'rb') as f:
        #     test_videos = pickle.load(f)

        # end_time = time.time()
        # print('Loading video takes: {:.2f}s\n'.format(end_time - start_time))

        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # transformer = transforms.Compose([
        #     transforms.RandomResizedCrop(224),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ColorJitter(.4, .4, .4),
        #     transforms.ToTensor(),
        #     normalize,
        # ])
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)) #((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))#(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        # if reset_seed is not None:
        #     seed_torch(reset_seed)

        dl_lst = []
        for seq in seq_lst:
            dataset = mm_dataset(claim, full_seq, seq, train_imgs, train_text, None, 
                sample_idx['train'], video_process_dir=video_process_dir, 
                video_fns_labels=video_fns_labels, mode='train', transforms=transformer, 
                norm_weights=norm_weights, 
                pick_sample_by_intsance_id=pick_sample_by_intsance_id,
                stop_overfitted_ins=stop_overfitted_ins, shuffle=True, split='train',
                seq_count=seq_count)
            dl_lst.append(DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True,
                                     collate_fn= mm_collate_fn, drop_last=drop_last_tr))

        # val_tfms = transforms.Compose([
        #     transforms.Resize(int(224 * 1.14)),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     normalize,
        # ])
        val_tfms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        
        val_dl_lst = []
        for seq in seq_lst:
            val_dataset = mm_dataset(claim, full_seq, seq, val_imgs, val_text, None, 
                sample_idx['val'], video_process_dir=video_process_dir, 
                video_fns_labels=video_fns_labels, mode='val', transforms=val_tfms, 
                norm_weights=norm_weights, split='val', seq_count=seq_count)
            val_dl_lst.append(DataLoader(val_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True,
                                     collate_fn= mm_collate_fn, drop_last=False))

        test_dl_lst = []
        for seq in seq_lst:
            test_dataset = mm_dataset(claim, full_seq, seq, test_imgs, test_text, None, 
                sample_idx['test'], video_process_dir=video_process_dir, 
                video_fns_labels=video_fns_labels, mode='test', transforms=val_tfms, 
                norm_weights=norm_weights, split='test', seq_count=seq_count)
            test_dl_lst.append(DataLoader(test_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True,
                                     collate_fn= mm_collate_fn, drop_last=False))

        # return itr, val_dataloader
        return dl_lst, val_dl_lst, test_dl_lst #[iter(cycle(dl)) for dl in dl_lst], val_dl_lst, test_dl_lst #[iter(cycle(dl)) for dl in ]

def mm_collate_fn(data):
    def selective_append(lst, x):
        if x is not None:
            lst.append(x)

    # the collate function for dataloader
    collate_images = []
    collate_text = []
    collate_videos = []
    collate_labels=[]
    collate_w=[]
    for d in data:
        selective_append(collate_images, d['img'])
        selective_append(collate_text, d['text'])
        selective_append(collate_videos, d['video'])
        selective_append(collate_labels, d['label'])
        selective_append(collate_w, d['sample_weight'])
        # collate_images.append(d['img'])
        # collate_text.append(d['text'])
        # # collate_videos.append(d['video'])
        # collate_labels.append((d['label']))
        # collate_w.append((d['sample_weight']))
    # collate_images = torch.stack(collate_images, dim=0)
    # collate_videos = torch.stack(collate_videos, dim=0)
    collate_labels=torch.tensor(collate_labels).reshape([-1,1])
    collate_w=torch.tensor(collate_w).reshape([-1,1])
    # return a dictionary of images and captions
    # return dict of collated images answers and questions
    return {
        'imgs': torch.stack(collate_images, dim=0) if len(collate_images) != 0 else None,
        'text': collate_text if len(collate_text) != 0 else None,
        'videos': torch.stack(collate_videos, dim=0) if len(collate_videos) != 0 else None,
        'labels': collate_labels,
        'sample_weights': collate_w
    }

# class bdd_dataset(Dataset):
#     def __init__(self, data_dir, full_seq, seq_idx, sample_idx, max_len=8, fps=3):
#         self.data_dir = data_dir
#         self.full_seq = full_seq
#         self.seq_idx = seq_idx
#         self.sample_idx = sample_idx
#         self.max_len = max_len
#         self.fps = fps

#         seq = self.full_seq[:seq_idx]
#         self.n_videos = seq.count('V')
#         self.n_gps = seq.count('G')

#     def __len__(self):
#         return self.len(self.sample_idx)

#     def __getitem__(self, idx):
#         video_name = self.sample_idx[idx]
#         # use end_time instead of start_time
#         video = read_video(self.data_dir, video_name)
#         gps = read_gps(self.data_dir, video_name)
#         labels = read_labels(self.data_dir, video_name)

#         video = video[-(self.max_len*self.fps-1):-1] # the last frame is for label
#         # consider truncating videos and save them to file to speed up loading
        
#         video_lst = [video[i*self.fps:(i+1)*self.fps] for i in range(self.n_videos)]
#         gps_lst = [gps[i*self.fps:(i+1)*self.fps] for i in range(self.n_gps)]

#         # end_frame_idx = n_videos * self.fps + 1
#         # end_gps_idx = n_gps * self.fps + 1

#         # video = video[:end_frame_idx]
#         # gps = gps[:end_gps_idx]
#         label = labels[-1]

#         return {'video': video_lst, 'gps': gps_lst, 'label': label}

# def bdd_batchgen(data_dir):
#     sample_idx = read()...
#     # tr_names, val_names = read_names(os.path.join(data_dir, 'sp'))
#     transforms = ...

#     for seq in sample_idx['train']:
#         bdd_dataset_tr = bdd_dataset(data_dir, seq, sample_idx['train'][seq])

#     transforms_val = ...
#     for seq in sample_idx['val']:
#         bdd_dataset_val = bdd_dataset(data_dir, seq, sample_idx['val'][seq])
    
#     return 

# def bdd_collate_fn(data):
#     collate_videos = []
#     collate_gps = []
#     collate_labels = []

#     for d in data:
#         if len(d['video']) != 0:
#             collate_videos.append(d['video']) 
#         if len(d['gps']) != 0:
#             collate_gps.append(d['gps'])
#         collate_labels.append(d['label'])
    
#     collate_labels = torch.tensor(collate_labels).reshape([-1,1])

#     return {
#         'videos': collate_videos if len(collate_videos) != 0 else None,
#         'gps': collate_gps if len(collate_gps) != 0 else None,
#         'labels': collate_labels
#     }

class mm_vqa_dataset(Dataset):
    def __init__(self, ques_file, ann_file, image_dir,vocab_file, transforms=None):
        vqa = VQA(annotation_file=ann_file, question_file=ques_file)
        self.imgs = []
        self.ques = []
        self.ans = []
        with open(vocab_file,'rb') as f:
            ans_to_id,id_to_ans=pickle.loads(f.read())
        # load the questions
        ques = vqa.questions['questions']
        # for tracking progress
        # for every question
        for x in tqdm.tqdm(ques,'Loading VQA data to memory'):
            # get the path
            answer = vqa.loadQA(x['question_id'])
            m_a=answer[0]['multiple_choice_answer']
            if m_a in ans_to_id:
                img_file = os.path.join(image_dir, '%012d.jpg' % (x['image_id']))
                self.imgs.append(img_file)
                # get the vector representation
                words = x['question']
                self.ques.append(words)
                self.ans.append(ans_to_id[m_a])
        self.transform = transforms
        self.N=len(self.ques)
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        
        img = self.transforming(self.imgs[idx])

        # if idx == 0:
        #     print('image shape:', img.shape)

        img_set = np.empty((2, img.shape[0], img.shape[1], img.shape[2]), np.dtype('float32'))

        ques_set = []
        
        img_set[0] = img
        ques_set.append(self.ques[idx])
        ans = self.ans[idx]

        if idx + 1 < self.N:
            img_set[1] = self.transforming(self.imgs[idx+1])
            ques_set.append(self.ques[idx+1])

            # if idx % 2:
            #     ans = self.ans[idx+1]
        else:
            img_set[1] = self.transforming(self.imgs[idx])
            ques_set.append(self.ques[idx])
        
        
        # finally return dictionary of images, questions and answers
        # for a given index
        return {'img': torch.from_numpy(img_set), 'ques': ques_set, 'ans': ans, 'sample_weights': 0.1}

    def transforming(self, img):
        img = Image.open(img)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img).convert("RGB")
        return img

class mm_vqa_dataset1(Dataset):
    def __init__(self, ques_file, ann_file, image_dir,vocab_file, transforms=None):
        vqa = VQA(annotation_file=ann_file, question_file=ques_file)
        self.imgs = []
        self.ques = []
        self.ans = []
        with open(vocab_file,'rb') as f:
            ans_to_id,id_to_ans=pickle.loads(f.read())
        # load the questions
        ques = vqa.questions['questions']
        # for tracking progress
        # for every question
        for x in tqdm.tqdm(ques,'Loading VQA data to memory'):
            # get the path
            answer = vqa.loadQA(x['question_id'])
            m_a=answer[0]['multiple_choice_answer']
            if m_a in ans_to_id:
                img_file = os.path.join(image_dir, '%012d.jpg' % (x['image_id']))
                self.imgs.append(img_file)
                # get the vector representation
                words = x['question']
                self.ques.append(words)
                self.ans.append(ans_to_id[m_a])
        self.transform = transforms
        self.N=len(self.ques)
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img).convert("RGB")
        
        # if idx == 0:
        #     print('image shape:', img.shape)

        ans = self.ans[idx]

        # if idx + 1 < self.N and idx % 2:
        #     ans = self.ans[idx+1]

        # finally return dictionary of images, questions and answers
        # for a given index
        return {'img': img, 'ques': [], 'ans': ans, 'sample_weights': 0.4}

class mm_vqa_dataset2(Dataset):
    def __init__(self, ques_file, ann_file, image_dir,vocab_file, transforms=None):
        vqa = VQA(annotation_file=ann_file, question_file=ques_file)
        self.imgs = []
        self.ques = []
        self.ans = []
        with open(vocab_file,'rb') as f:
            ans_to_id,id_to_ans=pickle.loads(f.read())
        # load the questions
        ques = vqa.questions['questions']
        # for tracking progress
        # for every question
        for x in tqdm.tqdm(ques,'Loading VQA data to memory'):
            # get the path
            answer = vqa.loadQA(x['question_id'])
            m_a=answer[0]['multiple_choice_answer']
            if m_a in ans_to_id:
                img_file = os.path.join(image_dir, '%012d.jpg' % (x['image_id']))
                self.imgs.append(img_file)
                # get the vector representation
                words = x['question']
                self.ques.append(words)
                self.ans.append(ans_to_id[m_a])
        self.transform = transforms
        self.N=len(self.ques)
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        img = self.transforming(self.imgs[idx])
        # if idx == 0:
        #     print('image shape:', img.shape)


        img_set = np.empty((2, img.shape[0], img.shape[1], img.shape[2]), np.dtype('float32'))

        ques_set = []
        
        img_set[0] = img
        ans = self.ans[idx]

        if idx + 1 < self.N:
            img_set[1] = self.transforming(self.imgs[idx+1])

            # if idx % 2:
            #     ans = self.ans[idx+1]
        else:
            img_set[1] = self.transforming(self.imgs[idx])
        
        
        # finally return dictionary of images, questions and answers
        # for a given index
        return {'img': torch.from_numpy(img_set), 'ques': ques_set, 'ans': ans, 'sample_weights': 0.3}

    def transforming(self, img):
        img = Image.open(img)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img).convert("RGB")
        return img


class mm_vqa_dataset3(Dataset):
    def __init__(self, ques_file, ann_file, image_dir,vocab_file, transforms=None):
        vqa = VQA(annotation_file=ann_file, question_file=ques_file)
        self.imgs = []
        self.ques = []
        self.ans = []
        with open(vocab_file,'rb') as f:
            ans_to_id,id_to_ans=pickle.loads(f.read())
        # load the questions
        ques = vqa.questions['questions']
        # for tracking progress
        # for every question
        for x in tqdm.tqdm(ques,'Loading VQA data to memory'):
            # get the path
            answer = vqa.loadQA(x['question_id'])
            m_a=answer[0]['multiple_choice_answer']
            if m_a in ans_to_id:
                img_file = os.path.join(image_dir, '%012d.jpg' % (x['image_id']))
                self.imgs.append(img_file)
                # get the vector representation
                words = x['question']
                self.ques.append(words)
                self.ans.append(ans_to_id[m_a])
        self.transform = transforms
        self.N=len(self.ques)
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        img = self.transforming(self.imgs[idx])
        # if idx == 0:
        #     print('image shape:', img.shape)

        img_set = np.empty((2, img.shape[0], img.shape[1], img.shape[2]), np.dtype('float32'))

        ques_set = []
        
        img_set[0] = img
        ques_set.append(self.ques[idx])
        ans = self.ans[idx]

        if idx + 1 < self.N:
            img_set[1] = self.transforming(self.imgs[idx+1])

            # if idx % 2:
            #     ans = self.ans[idx+1]
        else:
            img_set[1] = self.transforming(self.imgs[idx])
        
        # finally return dictionary of images, questions and answers
        # for a given index
        return {'img': torch.from_numpy(img_set), 'ques': ques_set, 'ans': ans, 'sample_weights': 0.2}

    def transforming(self, img):
        img = Image.open(img)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img).convert("RGB")
        return img

def mm_vqa_batchgen(vqa_dir, image_dir, num_workers=0, batch_size=1):
        # a transformation for the images
        vqa_train_ques=os.path.join(vqa_dir,'v2_OpenEnded_mscoco_train2014_questions.json')
        vqa_train_ann=os.path.join(vqa_dir,'v2_mscoco_train2014_annotations.json')
        vqa_val_ques=os.path.join(vqa_dir,'v2_OpenEnded_mscoco_val2014_questions.json')
        vqa_val_ann=os.path.join(vqa_dir,'v2_mscoco_val2014_annotations.json')
        vocab_file=os.path.join('conf/vqa_vocab.pkl')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transformer = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.4, .4, .4),
            transforms.ToTensor(),
            normalize,
        ])
        # the dataset
        dataset = mm_vqa_dataset(vqa_train_ques, vqa_train_ann, image_dir, vocab_file, transforms=transformer)
        dataset1 = mm_vqa_dataset1(vqa_train_ques, vqa_train_ann, image_dir, vocab_file, transforms=transformer)
        dataset2 = mm_vqa_dataset2(vqa_train_ques, vqa_train_ann, image_dir, vocab_file, transforms=transformer)
        dataset3 = mm_vqa_dataset3(vqa_train_ques, vqa_train_ann, image_dir, vocab_file, transforms=transformer)

        # the data loader
        dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True,
                                     collate_fn= mm_vqa_collate_fn, drop_last=True,pin_memory=True)
        dataloader1 = DataLoader(dataset1, num_workers=num_workers, batch_size=batch_size, shuffle=True,
                                     collate_fn= mm_vqa_collate_fn, drop_last=True,pin_memory=True)
        dataloader2 = DataLoader(dataset2, num_workers=num_workers, batch_size=batch_size, shuffle=True,
                                     collate_fn= mm_vqa_collate_fn, drop_last=True,pin_memory=True)
        dataloader3 = DataLoader(dataset3, num_workers=num_workers, batch_size=batch_size, shuffle=True,
                                     collate_fn= mm_vqa_collate_fn, drop_last=True,pin_memory=True)

        val_tfms = transforms.Compose([
            transforms.Resize(int(224 * 1.14)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        val_dataset = mm_vqa_dataset(vqa_val_ques, vqa_val_ann, image_dir, vocab_file, transforms=val_tfms)
        # the data loader
        val_dataloader = DataLoader(val_dataset, num_workers=num_workers, batch_size=int(batch_size/2), shuffle=True,
                                     collate_fn=mm_vqa_collate_fn, drop_last=False)
        # the iterator
        itr = iter(cycle(dataloader))
        itr1 = iter(cycle(dataloader1))
        itr2 = iter(cycle(dataloader2))
        itr3 = iter(cycle(dataloader3))
        # return itr, val_dataloader
        return itr,itr1,itr2,itr3,val_dataloader


def mm_vqa_collate_fn(data):
    # the collate function for dataloader
    collate_images = []
    collate_ques = []
    collate_ans=[]
    collate_w=[]
    for d in data:
        collate_images.append(d['img'])
        collate_ques.append(d['ques'])
        collate_ans.append((d['ans']))
        collate_w.append((d['sample_weights']))
    collate_images = torch.stack(collate_images, dim=0)
    collate_ans=torch.tensor(collate_ans).reshape([-1,1])
    collate_w=torch.tensor(collate_w).reshape([-1,1])
    # return a dictionary of images and captions
    # return dict of collated images answers and questions
    return {
        'img': collate_images,
        'ques': collate_ques,
        'ans': collate_ans,
        'sample_weights': collate_w
    }

class birds_dataset(Dataset):
    def __init__(self, data_dir, image_ids, image_dirs, labels, struct_features, bdng_box=None, transforms=None):
        self.imgs = []
        self.labels = []
        self.struct = []
        self.bdng_box = []

        self.N=len(image_ids)
        
        for i in tqdm.tqdm(range(self.N),'Loading birds data to memory'):
            img_id = image_ids[i]
            img_fn = os.path.join(data_dir, 'images', image_dirs[img_id])
            self.imgs.append(img_fn)
            self.labels.append(labels[img_id]-1)
            self.struct.append(struct_features[img_id])

            if bdng_box is not None:
                self.bdng_box.append(bdng_box[img_id])

        # self.labels = np.array(self.labels)
        # self.one_hot_labels = np.zeros((self.N, 200))
        # self.one_hot_labels[np.arange(self.N), self.labels.astype(int)] = 1
        # self.labels = [torch.from_numpy(np.asarray(x, dtype=int)) for x in list(self.one_hot_labels)]

        # self.struct = [torch.from_numpy(np.asarray(x, dtype=int)) for x in list(self.struct)]
        # self.struct = [torch.from_numpy(x).float() for x in list(self.struct)]

        self.transform = transforms
        # self.bdng_box = bdng_box
        
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        img = img.convert('RGB')
        if self.bdng_box is not None:
            x, y, width, height = self.bdng_box[idx]
            img = transforms.functional.resized_crop(img, y, x, height, width, (224,224))
            # img = transforms.ToTensor()(img)
            # print('cropped', img.shape)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img).convert("RGB")
        # print(img.shape)
        label = torch.as_tensor(self.labels[idx])
        struct = torch.as_tensor(self.struct[idx]).float()
        # finally return dictionary of images, questions and answers
        # for a given index
        return {'img': img, 'label': label, 'struct': struct}


def birds_batchgen(data_dir, num_workers=0, batch_size=1, testing=False):
        # a transformation for the images
        tr_te_fn = os.path.join(data_dir, 'train_test_split.txt')
        # tr_te_fn = os.path.join('/home/yxue/research/allstate/tmp_data', 'train_test_split.txt')
        tr_te = np.loadtxt(tr_te_fn)
        tr_ids = np.where(tr_te[:,1] == 1)[0] + 1
        te_ids = np.where(tr_te[:,1] == 0)[0] + 1

        if not testing:
            val_ids = tr_ids[np.random.choice(len(tr_ids), 1000, replace=False)]
            train_ids = np.array(list(set(tr_ids)-set(val_ids)))
        else:
            val_ids = []
            train_ids = tr_ids

        image_dirs = dict([
            (int(line.rstrip('\n').split(' ')[0]), line.rstrip('\n').split(' ')[1]) 
                for line in open(os.path.join(data_dir, 'images.txt'))])
        # image_dirs = dict([
        #     (int(line.rstrip('\n').split(' ')[0]), line.rstrip('\n').split(' ')[1]) 
        #         for line in open(os.path.join('/home/yxue/research/allstate/tmp_data', 'images.txt'))])
                    

        labels = dict([map(int, line.rstrip('\n').split(' ')) for line in open(os.path.join(data_dir, 'image_class_labels.txt'))])
        # labels = dict([map(int, line.rstrip('\n').split(' ')) for line in open(os.path.join('/home/yxue/research/allstate/tmp_data', 'image_class_labels.txt'))])

        df = pd.read_csv(os.path.join(data_dir, 'attributes/images_by_attributes.csv'), index_col=0)
        # df = pd.read_csv(os.path.join('/home/yxue/research/allstate/tmp_data', 'attributes/images_by_attributes.csv'), index_col=0)
        struct_features = dict(zip(range(1,df.shape[0]+1), df.values))

        bdng_box = dict(
            map(lambda x: (x[0], x[1:]), 
                [list(map(int, map(float, 
                    line.rstrip('\n').split(' ')))) 
                for line in open(os.path.join(data_dir, 'bounding_boxes.txt'))]))
        # bdng_box = dict(
        #     map(lambda x: (x[0], x[1:]), 
        #         [list(map(int, map(float, 
        #             line.rstrip('\n').split(' ')))) 
        #         for line in open(os.path.join('/home/yxue/research/allstate/tmp_data', 'bounding_boxes.txt'))]))


        # tr_image_dirs = image_dirs[tr_ids]
        # te_image_dirs = image_dirs[te_ids]

        # vqa_train_ques=os.path.join(vqa_dir,'v2_OpenEnded_mscoco_train2014_questions.json')
        # vqa_train_ann=os.path.join(vqa_dir,'v2_mscoco_train2014_annotations.json')
        # vqa_val_ques=os.path.join(vqa_dir,'v2_OpenEnded_mscoco_val2014_questions.json')
        # vqa_val_ann=os.path.join(vqa_dir,'v2_mscoco_val2014_annotations.json')
        # vocab_file=os.path.join('conf/vqa_vocab.pkl')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transformer = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(.4, .4, .4),
            transforms.ToTensor(),
            normalize,
        ])
        # the dataset
        dataset = birds_dataset(data_dir, train_ids, image_dirs, labels, struct_features, bdng_box=bdng_box, transforms=transformer)
        # the data loader
        dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True,
                                     collate_fn=birds_collate_fn, drop_last=True,pin_memory=True)
        val_tfms = transforms.Compose([
            transforms.Resize(int(224 * 1.14)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        val_dataset = birds_dataset(data_dir, val_ids, image_dirs, labels, struct_features, bdng_box=bdng_box, transforms=val_tfms)
        # the data loader
        val_dataloader = DataLoader(val_dataset, num_workers=num_workers, batch_size=int(batch_size/2), shuffle=True,
                                     collate_fn=birds_collate_fn, drop_last=False)
        
        test_dataset = birds_dataset(data_dir, te_ids, image_dirs, labels, struct_features, bdng_box=bdng_box, transforms=val_tfms)
        # the data loader
        test_dataloader = DataLoader(test_dataset, num_workers=num_workers, batch_size=int(batch_size/2), shuffle=True,
                                     collate_fn=birds_collate_fn, drop_last=False)
        
        # the iterator
        itr = iter(cycle(dataloader))
        return itr, val_dataloader, test_dataloader


def birds_collate_fn(data):
    # the collate function for dataloader
    collate_images = []
    collate_labels = []
    collate_struct = []
    for d in data:
        collate_images.append(d['img'])
        collate_labels.append(d['label'])
        collate_struct.append((d['struct']))
    collate_images = torch.stack(collate_images, dim=0)
    collate_labels = torch.stack(collate_labels, dim=0)
    collate_struct = torch.stack(collate_struct, dim=0)
    # return a dictionary of images and captions
    # return dict of collated images answers and questions
    return {
        'img': collate_images,
        'label': collate_labels,
        'struct': collate_struct
    }


# class vqa_struct_dataset(Dataset):
#     def __init__(self, ques_file, ann_file, image_dir,vocab_file, transforms=None):
#         vqa = VQA(annotation_file=ann_file, question_file=ques_file)
#         self.imgs = []
#         self.ques = []
#         self.ans = []
#         self.struct = []
#         with open(vocab_file,'rb') as f:
#             ans_to_id,id_to_ans=pickle.loads(f.read())
#         # load the questions
#         ques = vqa.questions['questions']
#         # for tracking progress
#         # for every question
#         for x in tqdm.tqdm(ques,'Loading VQA struct data to memory'):
#             # get the path
#             answer = vqa.loadQA(x['question_id'])
#             m_a=answer[0]['multiple_choice_answer']
#             if m_a in ans_to_id:
#                 img_file = os.path.join(image_dir, '%012d.jpg' % (x['image_id']))
#                 self.imgs.append(img_file)
#                 # get the vector representation
#                 words = x['question']
#                 self.ques.append(words)
#                 self.ans.append(ans_to_id[m_a])
#                 # self.struct.append(torch.empty(512).uniform_(0,1))
#                 self.struct.append(ans_to_id[m_a])
#         self.transform = transforms
#         self.N=len(self.ques)
#     def __len__(self):
#         return self.N

#     def __getitem__(self, idx):
#         img = Image.open(self.imgs[idx])
#         img = img.convert('RGB')
#         if self.transform is not None:
#             img = self.transform(img)
#         else:
#             img = transforms.ToTensor()(img).convert("RGB")
#         ques = self.ques[idx]
#         ans = self.ans[idx]
#         struct = self.struct[idx]
#         # finally return dictionary of images, questions and answers
#         # for a given index
#         return {'img': img, 'ques': ques, 'ans': ans, 'struct': struct}


# def vqa_struct_batchgen(vqa_dir, image_dir, num_workers=0, batch_size=1):
#         # a transformation for the images
#         vqa_train_ques=os.path.join(vqa_dir,'v2_OpenEnded_mscoco_train2014_questions.json')
#         vqa_train_ann=os.path.join(vqa_dir,'v2_mscoco_train2014_annotations.json')
#         vqa_val_ques=os.path.join(vqa_dir,'v2_OpenEnded_mscoco_val2014_questions.json')
#         vqa_val_ann=os.path.join(vqa_dir,'v2_mscoco_val2014_annotations.json')
#         vocab_file=os.path.join('conf/vqa_vocab.pkl')
#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         transformer = transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ColorJitter(.4, .4, .4),
#             transforms.ToTensor(),
#             normalize,
#         ])
#         # the dataset
#         dataset = vqa_struct_dataset(vqa_train_ques, vqa_train_ann, image_dir, vocab_file, transforms=transformer)
#         # the data loader
#         dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True,
#                                      collate_fn= vqa_struct_collate_fn, drop_last=True,pin_memory=True)
#         val_tfms = transforms.Compose([
#             transforms.Resize(int(224 * 1.14)),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             normalize,
#         ])
#         val_dataset = vqa_struct_dataset(vqa_val_ques, vqa_val_ann, image_dir, vocab_file, transforms=val_tfms)
#         # the data loader
#         val_dataloader = DataLoader(val_dataset, num_workers=num_workers, batch_size=int(batch_size/2), shuffle=True,
#                                      collate_fn=vqa_struct_collate_fn, drop_last=False)
#         # the iterator
#         itr = iter(cycle(dataloader))
#         return itr,val_dataloader


# def vqa_struct_collate_fn(data):
#     # the collate function for dataloader
#     collate_images = []
#     collate_ques = []
#     collate_ans=[]
#     collate_struct = []
#     for d in data:
#         collate_images.append(d['img'])
#         collate_ques.append(d['ques'])
#         collate_ans.append((d['ans']))
#         collate_struct.append((d['struct']))
#     collate_images = torch.stack(collate_images, dim=0)
#     collate_ans=torch.tensor(collate_ans).reshape([-1,1])
#     # return a dictionary of images and captions
#     # return dict of collated images answers and questions
#     return {
#         'img': collate_images,
#         'ques': collate_ques,
#         'ans': collate_ans,
#         'struct': collate_struct
#     }

class penn_dataset(Dataset):
    ''' Pytorch Penn Treebank Dataset '''

    def __init__(self, text_file,max_len=150):
        self.X = list()
        self.Y = list()
        with open(text_file) as f:
            # first line ignored as header
            f = f.readlines()[1:]
            for i in range(0,len(f),2):
                if len(f[i].split(' ', maxsplit=1)[1].split(' '))<max_len:
                    self.X.append(f[i])
                    self.Y.append(f[i+1])      
            assert len(self.X) == len(self.Y),\
            "mismatch in number of sentences & associated POS tags"
            self.count = len(self.X)
            del(f)
        
    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        return (self.X[idx].split(' ', maxsplit=1)[1],
                self.Y[idx].split()[1:])

    
class PennCollate:
    def __init__(self,vocab_file):
        with open(vocab_file,'r') as f:
            data=json.loads(f.read())
        self.tag_to_id=data['tag_to_id']
        self.id_to_tag=data['id_to_tag']
        
        
    def __call__(self,batch):
        pad_token=self.tag_to_id['<PAD>']
        text=[]
        tokens=[]
        max_len=0
        for b in batch:
            text.append(b[0].strip())
            tok=[self.tag_to_id[tag] for tag in b[1]]
            max_len=max(max_len,len(tok))
            tokens.append(tok)
        for i in range(len(tokens)):
            for j in range(max_len-len(tokens[i])):
                tokens[i].append(pad_token)
        tokens=torch.tensor(np.array(tokens))
        pad_mask=tokens.eq(pad_token)
        #Add padding to the tokens
        return {'text':text,'tokens':tokens,'pad_mask':pad_mask,'pad_id':pad_token}
    
def penn_dataloader(data_dir, batch_size=1, test_batch_size=1,num_workers=0,vocab_file='conf/penn_vocab.json'):
        train_file=os.path.join(data_dir,'train.txt')
        val_file=os.path.join(data_dir,'dev.txt')
        test_file=os.path.join(data_dir,'test.txt')
        collate_class=PennCollate(vocab_file)
        dataset=penn_dataset(train_file)
        val_dataset=penn_dataset(val_file)
        test_dataset=penn_dataset(test_file)
        train_dl=DataLoader(dataset,num_workers=num_workers,batch_size=batch_size,collate_fn=collate_class)
        val_dl=DataLoader(val_dataset,num_workers=num_workers,batch_size=test_batch_size,collate_fn=collate_class)
        test_dl=DataLoader(test_dataset,num_workers=num_workers,batch_size=test_batch_size,collate_fn=collate_class)
        train_dl=iter(cycle(train_dl))
        return train_dl,val_dl,test_dl
    




