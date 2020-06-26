import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import imageio as nd
from PIL import Image
from sklearn.model_selection import train_test_split

class _Getdata():
        
    def __init__(self, path= 'TinyImageNetDataset/', train= True, split=None):
    
        self.path = path
        self.train= train
        self.split = split

    def _get_id_dictionary(self):
        id_dict = {}
        for i, line in enumerate(open( self.path + 'wnids.txt', 'r')):
            id_dict[line.replace('\n', '')] = i
        return id_dict
    
    def _get_class_to_id_dict(self):
        id_dict = self._get_id_dictionary()
        all_classes = {}
        result = {}
        for i, line in enumerate(open( self.path + 'words.txt', 'r')):
            n_id, word = line.split('\t')[:2]
            all_classes[n_id] = word
        for key, value in id_dict.items():
            result[value] = (key, all_classes[key])      
        return result

    def _get_data(self):
        print('starting loading data')
        id_dict = self._get_id_dictionary()
        train_data, test_data = [], []
        train_labels, test_labels = [], []
        t = time.time()
        
        if self.train:
            
            for key, value in id_dict.items():
                train_data += [nd.imread( self.path + 'train/{}/images/{}_{}.JPEG'.format(key, key, str(i)), pilmode='RGB') for i in range(500)]
                train_labels_ = np.array([value]*500)
                train_labels += train_labels_.tolist()
            print('finished loading data, in {} seconds'.format(time.time() - t))
            
            if self.split:
                train_data, val_data, train_labels, val_labels = train_test_split(np.array(train_data), np.array(train_labels), test_size=(1-self.split), random_state=42)
                return train_data, train_labels, val_data, val_labels 
            
            return np.array(train_data), np.array(train_labels)

        else:
            
            for line in open(self.path + 'val/val_annotations.txt'):
                img_name, class_id = line.split('\t')[:2]
                test_data.append(nd.imread(self.path + 'val/images/{}'.format(img_name) ,pilmode='RGB'))
                test_labels.append(id_dict[class_id])
            print('finished loading data, in {} seconds'.format(time.time() - t))
            return np.array(test_data), np.array(test_labels)

    def get_classes(self):
        return  self._get_class_to_id_dict()


class TINDataset(Dataset):

    def __init__(self, data, labels, transform=None):

        self.transform = transform
        
        self.data, self.labels = data, labels
        self.data = self.data.transpose(0,3,1,2) #Transposing to match Tensor Image Format
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]

        if self.transform:
            x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1,2,0))
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)
