import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

data_dir = '/Users/ye/Documents/research/allstate/data/synthetic_mm_cifar_imdb_hmdb/'
out_fn = os.path.join(data_dir, 'sample_idx_ITTIV_pT.7_30k-10k-10k_adjusted_sample_sizes_seed10')

seed = 10

"""
cifar10: ndarray
  - default: 50,000 training and 10,000 test samples
  - selected classes:
    - (0) cat: 5000/1000
    - (1) dog: 5000/1000

imdb: 'imdb/train(test)/pos[(0)](neg[(1)])/id_star.txt'
  - default: 25,000 training and 25,000 test samples

hmdb: 'hmdb/classname/videonames'
  - default: 70/30 training/test split (1530 test samples; 30 per class)
  - selected classes:
    - (0) laugh: 70/28/30
    - (1) talk: 70/20/30
    
    
    
50:250:1
"""

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def random_samples(ids):
    np.random.shuffle(ids)
    return iter(cycle(ids))

def draw_sample(imgs, texts, videos, mode):
    if mode == 'I':
        return next(imgs)
    if mode == 'T':
        return next(texts)
    if mode == 'V':
        return next(videos)

def get_sample(seq, p, generators):
    sample = []
    for i,mode in enumerate(seq):
        if np.random.random() < p[mode]:
            sample.append(next(generators[i]))
        else:
            sample.append(-1)
    return sample

def get_sample_by_seq_type(seq, seq_name, generators):
    sample = []
    mode_cnt = dict(zip(['I','T','V'],list(map(int, seq_name.split('-')))))
    for i,mode in enumerate(seq):
        if mode_cnt[mode] > 0:
            sample.append(next(generators[i]))
            mode_cnt[mode] -= 1
        else:
            sample.append(-1)
    return sample

def get_sample_key(sample, seq):
    # sample = [1,2,3,-1,5]
    # seq    = 'ITTIV'
    sample = np.array(sample)
    seq = np.array(list(seq))
    img_ids = ','.join(map(str, sorted(sample[seq=='I'])))
    text_ids = ','.join(map(str,sorted(sample[seq=='T'])))
    video_ids = ','.join(map(str,sorted(sample[seq=='V'])))

    return ','.join([img_ids, text_ids, video_ids])


def create_synthetic_full_sequence_by_class_size(seq, img_labels, text_labels, video_labels, num_classes, p, sample_size_by_class, seed=10):
    np.random.seed(seed)
    samples = {}

    for cls in range(num_classes):
        img_ids = np.where(np.array(img_labels)==cls)[0]
        text_ids = np.where(np.array(text_labels)==cls)[0]
        video_ids = np.where(np.array(video_labels)==cls)[0]
        
        imgs = random_samples(img_ids)
        texts = random_samples(text_ids)
        videos = random_samples(video_ids)
        i = 0
        samples[cls] = []
        
        while True:
            samples[cls].append(get_sample(seq, p, imgs, texts, videos))
            i += 1

            if i >= sample_size_by_class[cls]:
                break
    
    return samples

def create_synthetic_full_sequence(seq, img_labels, text_labels, video_labels, num_classes, p, sample_size_by_class, sample_size_by_seq, seed=10):
    """ function to create synthetic data
    
    Args:
        seq (str): the sequence of modalities. 'I' stands for images, 'T' stands
            for text and 'V' for videos
        class_sizes (dict): nested dictionaries of class sizes of real datasets. 
            E.g dict({'I': dict({0: 6000, 1: 6000}), 'T': dict({0: 100, 1: 100})})
    Returns:
        A dict of samples by class
    """
    np.random.seed(seed)
    samples = {}

    for cls in range(num_classes):
        img_ids = np.where(np.array(img_labels)==cls)[0]
        text_ids = np.where(np.array(text_labels)==cls)[0]
        video_ids = np.where(np.array(video_labels)==cls)[0]
        
        generators = [random_samples(img_ids), random_samples(text_ids), 
                      random_samples(text_ids[1:]), random_samples(img_ids[1:]),
                      random_samples(video_ids)]
        samples[cls] = []
        
        sample_sizes = sample_size_by_seq.copy()
        
        for seq_name in sample_sizes:
            while sample_sizes[seq_name] > 0:
                sample = get_sample_by_seq_type(seq, seq_name, generators)
                samples[cls].append(sample)
                sample_sizes[seq_name] -= 1
                
    return samples

def create_synthetic_data(full_seq_with_label, seq):
    data = []
    for cls, d in full_seq_with_label.items():
        sample_keys = set([])
        for i, sample in enumerate(d):
            for t in range(len(seq)):
                if sample[t] != -1:
                    partial_sample = sample[:t+1]+[-1 for j in range(len(seq)-t-1)]
                    sample_key = get_sample_key(partial_sample, seq)
                    if sample_key in sample_keys:
                        continue
                    sample_keys.add(sample_key)
                    data.append([i,t,cls]+partial_sample)
    return data

def get_seq_name(sample):
    seq = list(map(get_mode_name, enumerate(sample)))
    return '%s-%s-%s'%(seq.count('I'),seq.count('T'),seq.count('V'))
    
def get_mode_name(x):
    if x[1] == -1:
        return '?'
    else:
        return full_seq[x[0]]

def group_by_seq_type(data):
    data_by_seq_type = {}
    for full_sample in data:
        sample = full_sample[-len(full_seq):]
        seq_type = get_seq_name(sample)
        
        if seq_type in data_by_seq_type:
            data_by_seq_type[seq_type].append(full_sample)
        else:
            data_by_seq_type[seq_type] = [full_sample]
    
    return data_by_seq_type

with open('/Users/ye/Documents/research/allstate/data/cifar2/cats_and_dogs_TrValTe', 'rb') as f:
    cats_and_dogs = pickle.load(f)

with open('/Users/ye/Documents/research/allstate/data/aclImdb/train_text.dict', 'rb') as fp:
    train_text = pickle.load(fp)
with open('/Users/ye/Documents/research/allstate/data/aclImdb/val_text.dict', 'rb') as fp:
    val_text = pickle.load(fp)
with open('/Users/ye/Documents/research/allstate/data/aclImdb/test_text.dict', 'rb') as fp:
    test_text = pickle.load(fp)

with open('/Users/ye/Documents/research/allstate/data/hmdb2/video_fns_labels', 'rb') as fp:
    video_data = pickle.load(fp)


pT = 0.7
full_seq = 'ITTIV'

def gen_data(pT):
    p = {'I': pT/5, 'T': pT, 'V': pT/12.5} # missing rate
    train_data = create_synthetic_full_sequence_by_class_size(full_seq, cats_and_dogs['train']['labels'], 
                                                train_text['labels'],
                                                video_data['train']['labels'],
                                                2, p, {0: 30000, 1:30000}, seed=seed)
    val_data = create_synthetic_full_sequence_by_class_size(full_seq, cats_and_dogs['val']['labels'], 
                                                val_text['labels'], 
                                                video_data['val']['labels'],
                                                2, p, {0: 10000, 1:10000}, seed=seed)
    test_data = create_synthetic_full_sequence_by_class_size(full_seq, cats_and_dogs['test']['labels'], 
                                               test_text['labels'], 
                                               video_data['test']['labels'],
                                               2, p, {0: 10000, 1:10000}, seed=seed)
#     train_data_timeline = create_synthetic_data(train_data, full_seq)
#     val_data_timeline = create_synthetic_data(val_data, full_seq)
#     test_data_timeline = create_synthetic_data(test_data, full_seq)


#     train_data_by_seq = group_by_seq_type(train_data_timeline)
#     val_data_by_seq = group_by_seq_type(val_data_timeline)
#     test_data_by_seq = group_by_seq_type(test_data_timeline)
    
    train_data_by_seq = group_by_seq_type(train_data[0])
    val_data_by_seq = group_by_seq_type(val_data[0])
    test_data_by_seq = group_by_seq_type(test_data[0])
    
    print([(k,len(v)) for k,v in train_data_by_seq.items()])
    print([(k,len(v)) for k,v in test_data_by_seq.items()])
    print('# tasks < 50 samples: ', sum(np.array([len(v) for _,v in test_data_by_seq.items()])<50))
    print('# tasks < 100 samples: ', sum(np.array([len(v) for _,v in test_data_by_seq.items()])<100))
    print('# tasks >= 200 samples: ', sum(np.array([len(v) for _,v in test_data_by_seq.items()])>=200))
    print('# tasks >= 300 samples: ', sum(np.array([len(v) for _,v in test_data_by_seq.items()])>=300))
    

# Run create_synthetic_full_sequence_by_class_size 
# first to get a sense of how many samples each seq type should have.
# Then manually increase sample sizes for seq with less than 100 samples.
# Run create_synthetic_full_sequence by specifying sample sizes per seq

sample_size_by_seq_tr = dict([('0-2-0', 10246), ('0-0-0', 1822), ('0-1-0', 8789), ('1-2-0', 3316), 
                              ('0-0-1', 30), ('1-0-0', 637), ('1-1-0', 2886), ('0-1-1', 535), ('0-2-1', 625), 
                              ('2-1-0', 234), ('2-2-0', 308), ('2-0-0', 50), ('1-1-1', 163), ('1-2-1', 213), 
                              ('1-0-1', 50), ('2-2-1', 50), ('2-1-1', 50), ('2-0-1', 50)])
sample_size_by_seq_val = dict([('0-2-0', 3441), ('0-1-0', 2911), ('0-0-0', 613), ('1-1-0', 982), ('1-2-0', 1089), 
                              ('0-2-1', 191), ('0-1-1', 165), ('1-0-0', 211), ('1-2-1', 87), ('2-1-0', 79), 
                              ('2-2-0', 102), ('1-1-1', 54), ('0-0-1', 20), ('2-2-1', 50), ('2-0-0', 50), 
                              ('1-0-1', 50), ('2-1-1', 50), ('2-0-1', 50)])
sample_size_by_seq_te = dict([('0-2-0', 3441), ('0-1-0', 2911), ('0-0-0', 613), ('1-1-0', 982), ('1-2-0', 1089), 
                              ('0-2-1', 191), ('0-1-1', 165), ('1-0-0', 211), ('1-2-1', 87), ('2-1-0', 79), 
                              ('2-2-0', 102), ('1-1-1', 54), ('0-0-1', 30), ('2-2-1', 50), ('2-0-0', 50), 
                              ('1-0-1', 50), ('2-1-1', 50), ('2-0-1', 50)])

def gen_data_by_sample_sizes_per_seq(pT):
    p = {'I': pT/5, 'T': pT, 'V': pT/12.5} # missing rate
    train_data = create_synthetic_full_sequence(full_seq, cats_and_dogs['train']['labels'], 
                                                train_text['labels'],
                                                video_data['train']['labels'],
                                                2, p, {0: 30000, 1:30000}, sample_size_by_seq=sample_size_by_seq_tr, seed=seed)
    val_data = create_synthetic_full_sequence(full_seq, cats_and_dogs['val']['labels'], 
                                                val_text['labels'], 
                                                video_data['val']['labels'],
                                                2, p, {0: 10000, 1:10000}, sample_size_by_seq=sample_size_by_seq_val, seed=seed)
    test_data = create_synthetic_full_sequence(full_seq, cats_and_dogs['test']['labels'], 
                                               test_text['labels'], 
                                               video_data['test']['labels'],
                                               2, p, {0: 10000, 1:10000}, sample_size_by_seq=sample_size_by_seq_te, seed=seed)
    
    
    train_data_timeline = create_synthetic_data(train_data, full_seq)
    val_data_timeline = create_synthetic_data(val_data, full_seq)
    test_data_timeline = create_synthetic_data(test_data, full_seq)


    train_data_by_seq = group_by_seq_type(train_data_timeline)
    val_data_by_seq = group_by_seq_type(val_data_timeline)
    test_data_by_seq = group_by_seq_type(test_data_timeline)
    
    print([(k,len(v)) for k,v in train_data_by_seq.items()])
    print([(k,len(v)) for k,v in val_data_by_seq.items()])
    print([(k,len(v)) for k,v in test_data_by_seq.items()])
    
    return {'train': train_data_by_seq, 'val':val_data_by_seq, 'test': test_data_by_seq}

sample_idx = gen_data_by_sample_sizes_per_seq(0.7) # with removing dup

with open(out_fn, 'wb') as fp:
    pickle.dump(sample_idx, fp)