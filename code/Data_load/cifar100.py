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
#                                 cifar100                                    #
#                                                                             #
###############################################################################


######################################### train #########################################


class cifar100_dataset(Data.Dataset):
    def __init__(self, train=True, 
                 transform=None, target_transform=None, 
                 split_percentage=0.9, random_seed=1, 
                 args=None,logger=None,num_class=100,
                 EM=False
                 ):
        
        # random_seed: t + args.seed

        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        original_images = np.load('data/cifar100/train_images.npy')
        original_labels = np.load('data/cifar100/train_labels.npy')

        print(" ")
        print("----------------------- Cifar100 -----------------------")
        print('shape of original images:', original_images.shape)

        ######## generate noisy labels ########
        data = torch.from_numpy(original_images).float()
        targets = torch.from_numpy(original_labels)
        dataset = zip(data, targets)

        if args.annotator_type=='instance-dependent':

            logger.info('Cifar100: Getting instance-dependent annotations......')
            file_name = 'data/cifar100_' + str(args.R) + '_' + str(args.l) + '/' + args.error_rate_type + '_' + str(random_seed)
            annotations = np.load(file_name + '_annotations.npy')
            noisy_label = np.load(file_name + '_noisy_label.npy')
            transition_true = np.load(file_name + '_transition_true.npy')
        
        else:
            logger.info('Wrong choice')
        

        ######## split: train and validation ########

        num_samples = int(original_images.shape[0])
        rng = default_rng(random_seed)
        train_set_index = rng.choice(num_samples, int(num_samples * split_percentage), replace=False)
        index_all = np.arange(num_samples)
        val_set_index = np.delete(index_all, train_set_index)

        # image
        self.train_data = original_images[train_set_index]
        self.val_data = original_images[val_set_index]

        # true labels
        self.train_labels = original_labels[train_set_index]
        self.val_labels = original_labels[val_set_index]

        # trusted labels
        self.train_label_trusted_1 = -1 * np.ones(len(train_set_index))
        self.val_label_trusted_1 = -1 * np.ones(len(val_set_index))
        self.train_label_trusted_2 = -1 * np.ones(len(train_set_index))
        self.val_label_trusted_2 = -1 * np.ones(len(val_set_index))

        # noisy annotations (n, R)

        self.train_annotations = annotations[train_set_index]
        self.val_annotations = annotations[val_set_index]

        # noisy label (n, ): 
        """
            * Not specified: noisy label (mnist; cifar10; cifar100); majority vote labels (cifar10n; labelme; music)
            * EM == True: EM labels
        """

        self.train_noisy_label = noisy_label[train_set_index]
        self.val_noisy_label = noisy_label[val_set_index]
        
        if EM == True:
            EM_labels = utils.tools.DS_EM(annotations, original_labels, num_class, seed=random_seed, noisy_label=noisy_label)
            self.train_noisy_label = EM_labels[train_set_index]
            self.val_noisy_label = EM_labels[val_set_index]
        
        print('error rate (train):', (self.train_noisy_label != original_labels[train_set_index]).sum() / self.train_noisy_label.shape[0])
        print('error rate (val):', (self.val_noisy_label != original_labels[val_set_index]).sum() / self.val_noisy_label.shape[0])

        # transition
        if args.annotator_type=='instance-dependent':
            self.train_transition_true = transition_true[train_set_index]
            self.val_transition_true = transition_true[val_set_index]

        if train:
            print("shape of annotations (train)", self.train_annotations.shape)
        else:
            print("shape of annotations (validation)", self.val_annotations.shape)

        print("cifar10 dataset initialization: Done! \n")

	    
	
    def __getitem__(self, index):
        
        if self.train:
            img = self.train_data[index]
            label = self.train_labels[index]
            trusted_label_1 = self.train_label_trusted_1[index]
            trusted_label_2 = self.train_label_trusted_2[index]
            
            annot = self.train_annotations[index]
            noisy_label = self.train_noisy_label[index]
            			
        else:
            img = self.val_data[index]
            label = self.val_labels[index]
            trusted_label_1 = self.val_label_trusted_1[index]
            trusted_label_2 = self.val_label_trusted_2[index]

            annot = self.val_annotations[index]
            noisy_label = self.val_noisy_label[index]
        
        img = Image.fromarray(img)

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

    def get_true_transition(self, idx):
        if self.train:
            return torch.tensor(self.train_transition_true[idx])
        else:
            return torch.tensor(self.val_transition_true[idx])
    
    def get_annot(self, idx):
        if self.train:
            return torch.tensor(self.train_annotations[idx])
        else:
            return torch.tensor(self.val_annotations[idx])
    
    def get_noisy_label(self, idx):
        if self.train:
            return torch.tensor(self.train_noisy_label[idx])
        else:
            return torch.tensor(self.val_noisy_label[idx])
    
    def get_trusted_label_1(self, idx):
        if self.train:
            return torch.tensor(self.train_label_trusted_1[idx])
        else:
            return torch.tensor(self.val_label_trusted_2[idx])
    
    def get_trusted_label_2(self, idx):
        if self.train:
            return torch.tensor(self.train_label_trusted_2[idx])
        else:
            return torch.tensor(self.val_label_trusted_2[idx])


######################################### test #########################################

		
class cifar100_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):
        
        print(" ")
        print("----------------------- Cifar100 (test) -----------------------")

        self.transform = transform
        self.target_transform = target_transform
        
        self.test_data = np.load('data/cifar100/test_images.npy')
        self.test_labels = np.load('data/cifar100/test_labels.npy')


        print("test data shape:", self.test_data.shape)


    def __getitem__(self, index):
        
        img, label = self.test_data[index], self.test_labels[index]
        
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
    
        return img, label
    
    def __len__(self):
        return len(self.test_data)

