import os
import pickle
import numpy as np

data_dir = '/files/yxue/research/allstate/data/socialiq'
train_video_dir = data_dir+'_raw/train/vision/raw'

def split(vdir, odir):
    np.random.seed(918)
    video_names = []
    split_dict = {}
    for root, dirs, files in os.walk(vdir):
        for f in files:
            video_path = os.path.join(root,f)
            if f == 'deKPBy_uLkg_trimmed-out.mp4':
                continue

            video_names.append(f.split('.')[0])
    val_size = test_size = 100

    np.random.shuffle(video_names)

    for video_name in video_names[:test_size]:
        split_dict[video_name] = 'test'

    for video_name in video_names[test_size:(test_size+val_size)]:
        split_dict[video_name] = 'val'

    for video_name in video_names[(test_size+val_size):]:
        split_dict[video_name] = 'train'

    with open(os.path.join(odir,'split_with_test.dict.pkl'), 'wb') as f:
        pickle.dump(split_dict, f)

split(train_video_dir, data_dir+'/train')