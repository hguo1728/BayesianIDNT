import torch
import torch.utils.data as Data

import numpy as np
from numpy import genfromtxt
from numpy.matlib import repmat
from numpy.random import default_rng

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from PIL import Image
from scipy import stats
from pathlib import Path

from Data_load import transformer 
from utils import synthetic 
import utils.tools, pdb


###############################################################################
#                                                                             #
#                                 LabelMe                                     #
#                                                                             #
###############################################################################


######################################### train #########################################

class labelme_dataset(Data.Dataset):
    def __init__(self, train=True, 
                 transform=None, target_transform=None, 
                 split_percentage=0.9, random_seed=1, 
                 args=None,logger=None,num_class=8,
                 EM = False
                 ):
        
        print(" ")
        print("----------------------- LabelMe (train) -----------------------")
        
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.train_data = np.load('data/labelme/prepared/data_train_vgg16.npy')
        self.train_labels = np.load('data/labelme/prepared/labels_train.npy')
        self.val_data = np.load('data/labelme/prepared/data_valid_vgg16.npy')
        self.val_labels= np.load('data/labelme/prepared/labels_valid.npy')
        
        if self.train:

            # annotation
            if args.error_rate_type=='real':
                logger.info('LabelMe: Getting real annotations......')
                annotations = np.load('data/labelme/prepared/answers.npy')
                self.train_annotations = annotations.astype(int)
            else:
                logger.info('Wrong choice')
            
            # noisy label: majority vote or EM
            noisy_label = np.load('data/labelme/prepared/labels_train_mv.npy')

            if EM:
                # self.train_noisy_label = np.load('data/labelme/prepared/labels_train_DS.npy')
                EM_labels = utils.tools.DS_EM(annotations, self.train_labels, num_class, seed=random_seed, noisy_label=noisy_label)
                self.train_noisy_label = EM_labels
                
            else:
                self.train_noisy_label = noisy_label
        
        # trusted labels
        self.train_label_trusted_1 = -1 * np.ones(len(self.train_labels))
        self.val_label_trusted_1 = -1 * np.ones(len(self.val_labels))
        self.train_label_trusted_2 = -1 * np.ones(len(self.train_labels))
        self.val_label_trusted_2 = -1 * np.ones(len(self.val_labels))

        print('error rate (train):', (self.train_noisy_label != self.train_labels).sum() / self.train_noisy_label.shape[0])

        if train:
            print("shape of annotations (train)", self.train_annotations.shape)
        
        print("labelme dataset initialization: Done! \n")



    def __getitem__(self, index):
           
        if self.train:
            img = self.train_data[index]
            label = self.train_labels[index]
            trusted_label_1 = self.train_label_trusted_1[index]
            trusted_label_2 = self.train_label_trusted_1[index]
            
            annot = self.train_annotations[index]
            noisy_label = self.train_noisy_label[index]
            			
        else:
            img = self.val_data[index]
            label = self.val_labels[index]
            trusted_label_1 = self.val_label_trusted_1[index]
            trusted_label_2 = self.val_label_trusted_1[index]

            annot = -1
            noisy_label = -1

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)
            trusted_label_1 = self.target_transform(trusted_label_1)
            trusted_label_2 = self.target_transform(trusted_label_2)

            annot = self.target_transform(annot)
            noisy_label = self.target_transform(noisy_label)
        
        return index, img, label, trusted_label_1, trusted_label_2, annot, noisy_label

    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.val_data)
        
	
    def update_trusted_label_1(self, trusted_label, idx):
        trusted_label = torch.from_numpy(trusted_label).float()
        if self.train:
            self.train_label_trusted_1[idx] = trusted_label
        else:
            self.val_label_trusted_1[idx] = trusted_label
    
    def update_trusted_label_2(self, trusted_label, idx):
        trusted_label = torch.from_numpy(trusted_label).float()
        if self.train:
            self.train_label_trusted_2[idx] = trusted_label
        else:
            self.val_label_trusted_2[idx] = trusted_label

    
    def get_annot(self, idx):
        if self.train:
            return torch.tensor(self.train_annotations[idx])
        else:
            print("Wrong!")
            return
        
    def get_noisy_label(self, idx):
        if self.train:
            return torch.tensor(self.train_noisy_label[idx])
        else:
            print("Wrong!")
            return
    
    def get_trusted_label_1(self, idx):
        if self.train:
            return torch.tensor(self.train_label_trusted_1[idx])
        else:
            return torch.tensor(self.val_label_trusted_1[idx])
    
    def get_trusted_label_2(self, idx):
        if self.train:
            return torch.tensor(self.train_label_trusted_2[idx])
        else:
            return torch.tensor(self.val_label_trusted_2[idx])



######################################### test #########################################

        
class labelme_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None, test=True, return_idx = False):

        self.transform = transform
        self.target_transform = target_transform
        self.test = test
        self.return_idx = return_idx

        if self.test:

            print(" ")
            print("----------------------- LabelMe (test) -----------------------")
            
            self.test_data = np.load('data/labelme/prepared/data_test_vgg16.npy')
            self.test_labels= np.load('data/labelme/prepared/labels_test.npy')

            print("test data shape:", self.test_data.shape)
        
        else:

            print(" ")
            print("----------------------- LabelMe (validation) -----------------------")
            
            self.test_data = np.load('data/labelme/prepared/data_valid_vgg16.npy')
            self.test_labels= np.load('data/labelme/prepared/labels_valid.npy')

            print("validation data shape:", self.test_data.shape)



    def __getitem__(self, index):
        
        img, label = self.test_data[index], self.test_labels[index]
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        if self.return_idx:
            return index, img, label
        else:
            return img, label
    
    def __len__(self):
        return len(self.test_data)
       

        
