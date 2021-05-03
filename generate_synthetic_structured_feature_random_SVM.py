import numpy as np
import os
import pickle

save_iter = 200
n_cls = 10
# n_samples_per_cls = 84978
feature_size = 128
out_path = '/files/yxue/research/allstate/data/vqa'

for c in range(n_cls):
    os.makedirs(os.path.join(out_path, 'synthetic_structured_data_SVM3500/%s'%c))

with open('/files/yxue/research/allstate/data/vqa/class_sizes_tr', 'rb') as f:
    cls_sizes_tr = pickle.load(f)

with open('/files/yxue/research/allstate/data/vqa/class_sizes_val', 'rb') as f:
    cls_sizes_val = pickle.load(f)

cls_sizes = {}
for c in set(list(cls_sizes_tr.keys())+list(cls_sizes_val.keys())):
    cls_sizes[c] = 0
    if c in cls_sizes_tr:
        cls_sizes[c] += cls_sizes_tr[c]
    if c in cls_sizes_val:
        cls_sizes[c] += cls_sizes_val[c]

sorted_sizes = sorted(list(cls_sizes.values()))[::-1]
sorted_cls = sorted(cls_sizes, key=lambda k: d[k])[::-1]

cls_sizes = dict(zip(sorted_cls, sorted_sizes))

ws = np.random.uniform(low=-1,high=1,size=(n_cls,feature_size))
ws = ws/np.linalg.norm(ws, axis=1)[:,None]

cls_id = 0
cls_dict = {}
Y = {}

fn_idx = dict(zip(cls_sizes.keys(), [0]*len(cls_sizes)))

# Y = dict(zip(cls_sizes.keys(), [[0]*cls_sizes[i] for i in cls_sizes]))
# idx = dict(zip(cls_sizes.keys(), [0]*len(cls_sizes)))

# Y = dict(zip(range(n_cls), [[0]*cls_sizes[i] for i in range(n_cls)]))
# idx = dict(zip(range(n_cls), [0]*n_cls))

it = 0
while True:
    it += 1
    print('iter: ', it)
    if sum(cls_sizes.values()) == 0:
        break
        
    points = np.random.uniform(low=-10, high=10, size=(5000,feature_size)) #np.random.random((20,2))
    res = np.dot(points, ws.transpose())

    selected_idx = ((res>=1).sum(axis=1)==1) * ((res<=-1).sum(axis=1)==n_cls-1)
    selected_points = points[selected_idx]
    selected_cls = res[selected_idx].argmax(axis=1)

    for i in range(len(points)):
        c_raw = ''.join(map(lambda x: '1' if x else '0', res[i]))
        if c_raw not in cls_dict:
            cls_dict[c_raw] = cls_id
            cls_id += 1
        
        c = cls_dict[c_raw]
        
        if c in cls_sizes and cls_sizes[c] > 0:
            cls_sizes[c] -= 1

            if c in Y:
                Y[c].append(points[i])
            else:
                Y[c] = [points[i]]

    if it % save_iter == 0:
        for c in Y:
            if len(Y[c]) > 0:
                np.savetxt(os.path.join(out_path, 'synthetic_structured_data_SVM3500/%s/%s'%(c,fn_idx[c])), np.vstack(Y[c]))
                fn_idx[c] += 1

                Y[c] = []

for c in Y:
    if len(Y[c]) > 0:
        np.savetxt(os.path.join(out_path, 'synthetic_structured_data_SVM3500/%s/%s'%(c,fn_idx[c])), np.vstack(Y[c]))

print('done')
# with open(os.path.join(out_path, 'synthetic_structured_data_SVM3500.dict'), 'wb') as f:
#     pickle.dump(Y, f)
