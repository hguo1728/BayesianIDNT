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
#                            trusted dataset                                  #
#                                                                             #
###############################################################################

class trusted_dataset(Data.Dataset):
        def __init__(self, t, train=True, transform=None, target_transform=None, args=None):

            self.transform = transform
            self.target_transform = target_transform
            self.train = train

            if self.train:
                 self.train_data = np.load(args.log_folder + '/' + 'trial_' + str(t+1) + '/train_trusted_imgs.npy')
                 self.train_labels = np.load(args.log_folder + '/' + 'trial_' + str(t+1) + '/train_trusted_labels_true.npy')
                 self.train_labels_trusted = np.load(args.log_folder + '/' + 'trial_' + str(t+1) + '/train_trusted_labels.npy')

                 print(" ")
                 print("----------------------- Trusted data (Train) -----------------------")
                 print("training data shape (trusted data)", self.train_data.shape)

            else:
                 self.val_data = np.load(args.log_folder + '/' + 'trial_' + str(t+1) + '/val_trusted_imgs.npy')
                 self.val_labels = np.load(args.log_folder + '/' + 'trial_' + str(t+1) + '/val_trusted_labels_true.npy')
                 self.val_labels_trusted = np.load(args.log_folder + '/' + 'trial_' + str(t+1) + '/val_trusted_labels.npy')

                 print(" ")
                 print("----------------------- Trusted data (Validation) -----------------------")
                 print("validation data shape (trusted data)", self.val_data.shape)

            
        def __getitem__(self, index):

            if self.train:
                 img = self.train_data[index]
                 true_label = self.train_labels[index]
                 trusted_label = self.train_labels_trusted[index]
            else:
                 img = self.val_data[index]
                 true_label = self.val_labels[index]
                 trusted_label = self.val_labels_trusted[index]

            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                true_label = self.target_transform(true_label)
                true_label = self.target_transform(trusted_label)
                
            return index, img, true_label, trusted_label

        def __len__(self):
            if self.train:
                 return len(self.train_data)
            else:
                 return len(self.val_data)
        
        def update_trusted_data(self, img, label, trusted_label):
             if self.train:
                  self.train_data = img
                  self.train_labels = label
                  self.train_labels_trusted = trusted_label
             else:
                  self.val_data = img
                  self.val_labels = label
                  self.val_labels_trusted = trusted_label