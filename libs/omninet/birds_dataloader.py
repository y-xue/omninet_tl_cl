class birds_dataset(Dataset):
    def __init__(self, data_dir, image_ids, image_dirs, labels, struct_features, bdng_box=None, transforms=None):
        self.imgs = []
        self.labels = []
        self.struct = []

        self.N=len(image_ids)
        
        for i in tqdm.tqdm(range(self.N),'Loading VQA struct data to memory'):
            img_id = image_ids[i]
            img_fn = os.path.join(data_dir, 'images', image_dirs[img_id])
            self.imgs.append(img_fn)
            self.labels.append(labels[img_id])
            self.struct.append(struct_features[img_id])

        self.transform = transforms
        self.bdng_box = bdng_box
        
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        img = img.convert('RGB')
        x, y, width, height = self.bdng_box[idx]
        if self.bdng_box is not None:
            img = transforms.functional.resized_crop(img, y, x, height, width, 224)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img).convert("RGB")
        label = self.labels[idx]
        struct = self.struct[idx]
        # finally return dictionary of images, questions and answers
        # for a given index
        return {'img': img, 'label': label, 'struct': struct}


def birds_batchgen(data_dir, num_workers=0, batch_size=1):
        # a transformation for the images
        tr_te_fn = os.path.join(data_dir, 'train_test_split.txt')
        tr_te = np.loadtxt(tr_te_fn)
        tr_ids = np.where(tr_te[:,1] == 1)[0] + 1
        te_ids = np.where(tr_te[:,1] == 0)[0] + 1

        image_dirs = dict([
            (int(line.rstrip('\n').split(' ')[0]), line.rstrip('\n').split(' ')[1]) 
                for line in open(os.path.join(data_dir, 'images.txt'))])

        labels = dict([map(int, line.rstrip('\n').split(' ')) for line in open(os.path.join(data_dir, 'image_class_labels.txt'))])

        df = pd.read_csv(os.path.join(data_dir, 'attributes/images_by_attributes.csv'), index_col=0)
        struct_features = dict(zip(range(1,df.shape[0]+1), df.values))

        bdng_box = dict(
            map(lambda x: (x[0], x[1:]), 
                [list(map(int, map(float, 
                    line.rstrip('\n').split(' ')))) 
                for line in open(os.path.join(data_dir, 'bounding_boxes.txt'))]))
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
        dataset = birds_dataset(data_dir, tr_ids, image_dirs, labels, struct_features, bdng_box=bdng_box, transforms=transformer)
        # the data loader
        dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True,
                                     collate_fn=birds_collate_fn, drop_last=True,pin_memory=True)
        val_tfms = transforms.Compose([
            transforms.Resize(int(224 * 1.14)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        val_dataset = birds_dataset(data_dir, te_ids, image_dirs, labels, struct_features, bdng_box=bdng_box, transforms=val_tfms)
        # the data loader
        val_dataloader = DataLoader(val_dataset, num_workers=num_workers, batch_size=int(batch_size/2), shuffle=True,
                                     collate_fn=birds_collate_fn, drop_last=False)
        # the iterator
        itr = iter(cycle(dataloader))
        return itr,val_dataloader


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
    # return a dictionary of images and captions
    # return dict of collated images answers and questions
    return {
        'img': collate_images,
        'label': collate_labels,
        'struct': collate_struct
    }

class vqa_struct_dataset(Dataset):
    def __init__(self, ques_file, ann_file, image_dir,vocab_file, transforms=None):
        vqa = VQA(annotation_file=ann_file, question_file=ques_file)
        self.imgs = []
        self.ques = []
        self.ans = []
        self.struct = []
        with open(vocab_file,'rb') as f:
            ans_to_id,id_to_ans=pickle.loads(f.read())
        # load the questions
        ques = vqa.questions['questions']
        # for tracking progress
        # for every question
        for x in tqdm.tqdm(ques,'Loading VQA struct data to memory'):
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
                # self.struct.append(torch.empty(512).uniform_(0,1))
                self.struct.append(ans_to_id[m_a])
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
        ques = self.ques[idx]
        ans = self.ans[idx]
        struct = self.struct[idx]
        # finally return dictionary of images, questions and answers
        # for a given index
        return {'img': img, 'ques': ques, 'ans': ans, 'struct': struct}


def vqa_struct_batchgen(vqa_dir, image_dir, num_workers=0, batch_size=1):
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
        dataset = vqa_struct_dataset(vqa_train_ques, vqa_train_ann, image_dir, vocab_file, transforms=transformer)
        # the data loader
        dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True,
                                     collate_fn= vqa_struct_collate_fn, drop_last=True,pin_memory=True)
        val_tfms = transforms.Compose([
            transforms.Resize(int(224 * 1.14)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        val_dataset = vqa_struct_dataset(vqa_val_ques, vqa_val_ann, image_dir, vocab_file, transforms=val_tfms)
        # the data loader
        val_dataloader = DataLoader(val_dataset, num_workers=num_workers, batch_size=int(batch_size/2), shuffle=True,
                                     collate_fn=vqa_struct_collate_fn, drop_last=False)
        # the iterator
        itr = iter(cycle(dataloader))
        return itr,val_dataloader


def vqa_struct_collate_fn(data):
    # the collate function for dataloader
    collate_images = []
    collate_ques = []
    collate_ans=[]
    collate_struct = []
    for d in data:
        collate_images.append(d['img'])
        collate_ques.append(d['ques'])
        collate_ans.append((d['ans']))
        collate_struct.append((d['struct']))
    collate_images = torch.stack(collate_images, dim=0)
    collate_ans=torch.tensor(collate_ans).reshape([-1,1])
    # return a dictionary of images and captions
    # return dict of collated images answers and questions
    return {
        'img': collate_images,
        'ques': collate_ques,
        'ans': collate_ans,
        'struct': collate_struct
    }
