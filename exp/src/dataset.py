import numpy as np
from torch.utils.data.dataset import Dataset
import os
import torch
import pickle

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# 数据集读取 等待更改适配数据集
class Multimodal_Datasets(Dataset):
    # def __init__(self, dataset_path, data='mosei_senti', split_type='train', if_align=False):
    def __init__(self, dataset_path, data='MMHS150K+prompt', split_type='train', if_align=False):
        super(Multimodal_Datasets, self).__init__()
        # dataset_path = os.path.join(dataset_path, data+'_data.pkl' if if_align else data+'_data_noalign.pkl' )
        dataset_path = os.path.join(dataset_path, data + '.pkl')
        dataset = pickle.load(open(dataset_path, 'rb')) # 读取数据集

        # These are torch tensors vision对应img text对应tweet audio对应imgtext

        self.labels = torch.tensor(dataset[split_type]['label'].astype(np.float32)).cpu().detach()
        self.vision = torch.tensor(dataset[split_type]['img'].astype(np.float32)).cpu().detach()
        self.audio = torch.tensor(dataset[split_type]['docs_img'].astype(np.float32)).cpu().detach()
        self.text = torch.tensor(dataset[split_type]['docs_tweet'].astype(np.float32)).cpu().detach()
        self.prompt = torch.tensor(dataset[split_type]['prompt'].astype(np.float32)).cpu().detach()
        # Note: this is STILL an numpy array
        # self.meta = dataset[split_type]['id'] if 'id' in dataset[split_type].keys() else None

        self.data = data

        self.n_modalities = 4 # vision/ text/ audio
    def get_n_modalities(self):
        return self.n_modalities
    def get_seq_len(self):
        # return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1], self.prompt.shape[1]
    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2], self.prompt.shape[2]
    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return self.labels.shape[1], self.labels.shape[2]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        X = (index, self.text[index], self.audio[index], self.vision[index], self.prompt[index])
        # X = (index, self.text[index], self.audio[index], self.vision[index])
        Y = self.labels[index]
        Y = torch.argmax(Y, dim=-1)
        return X, Y

