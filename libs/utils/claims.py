class Claim:
    def __init__(self, seq, weights):
        self.seq = seq
        self.weights = weights

    def get_image_files(self, image_dir):
        return 0

    def get_images(self, image_dir, h=32, w=32):
        imgs = np.loadtxt(image_dir+'/measures',delimiter=',')
        imgs = imgs.reshape(-1,3,h,w).transpose((0,2,3,1))
        return imgs

    def read_txt(self, fn):
        with open(fn, 'r+', encoding="utf-8") as f:
            return f.read()

    def list_files(self, d):
        return [f for f in os.listdir(d) if os.path.isfile(os.path.join(d,f))]

    def get_text(self, text_dir):
        pos_files = self.list_files(text_dir+'/train/pos')
        neg_files = self.list_files(text_dir+'/train/neg')

        text = [self.read_txt(os.path.join(text_dir,'train/pos', fn)) for fn in pos_files]
        text.extend([self.read_txt(os.path.join(text_dir,'train/neg', fn)) for fn in neg_files])
        return text

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

    def get_weight(self, sample):
        # sample: [instance_id,t,c,...]
        return self.weights[int(sample[1])]