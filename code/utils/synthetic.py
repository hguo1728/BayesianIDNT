from __future__ import division
#import mxnet as mx
import numpy as np
from math import inf
import math
import logging,os
import copy
import urllib
import logging,os,sys
from random import shuffle

from scipy import stats
from scipy import special
from scipy.optimize import linear_sum_assignment
from scipy import stats
from scipy.stats import multivariate_normal
from scipy.special import softmax

import torch
from torch.nn import functional as F
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix

from statistics import multimode

from PIL import Image

from Data_load.transformer import *


################################################################################################
#                                                                                              #
#                         generate synthetic data with noisy labels                            #
#                                                                                              #
################################################################################################

"""

generate synthetic data from two 2D normal distributions:
positive examples: N(u, I)
negative examples: N(-u, I)
u = (-2, 2)'

generate noisy labels: Xia et al, 2020 Nips
IDN-10%; IDN-30%; IDN-50%

"""

def generate_synthetic_data_and_labels(seed, tau=[0.1, 0.3, 0.5], norm_std=0.1, intercept=False):
    """
    tau -> noise_rate [0.1, 0.3, 0.5]

    """

    print("building dataset...")
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))

    rng = np.random.default_rng(int(seed))

    num_classes = 2
    feature_dim = 2

    # -------------- generate data -------------

    # Positive
    center_1 = [-2, 2]
    cov_1 = np.eye(feature_dim)
    P_total = 1000
    P_train = 500
    P_test = P_total - P_train

    Positive = multivariate_normal.rvs(mean=center_1, cov=cov_1, size=P_total) # P_total * 2

    # negative
    center_2 = [2, -2]
    cov_2 = np.eye(feature_dim)
    N_total = 1000
    N_train = 500
    N_test = N_total - N_train

    Negative = multivariate_normal.rvs(mean=center_2, cov=cov_2, size=N_total) # out: N_total * featue_dim

    # train data
    train_num = P_train + N_train
    train_X = np.concatenate((Positive[:P_train, :], Negative[:N_train, :]), axis=0)
    train_Y_clean = np.concatenate((np.ones(P_train), np.zeros(N_train)))

    train = np.concatenate((train_X, train_Y_clean.reshape((train_num, 1))), axis=1)
    rng.shuffle(train, axis=0)

    train_X = train[:, :feature_dim]
    train_Y_clean = train[:, -1]


    # test data
    test_num = P_test + N_test
    test_X = np.concatenate((Positive[P_train:, :], Negative[N_train:, :]), axis=0)
    test_Y = np.concatenate((np.ones(P_test), np.zeros(N_test)))

    test = np.concatenate((test_X, test_Y.reshape((test_num, 1))), axis=1)
    rng.shuffle(test, axis=0)

    test_X = test[:, :feature_dim]
    test_Y = test[:, -1]


    # intercept

    if intercept:
        train_X = np.concatenate((np.ones((train_num, 1)), train_X), axis=1)
        test_X = np.concatenate((np.ones((test_num, 1)), test_X), axis=1)

        feature_dim += 1

    # ------------- generate noisy labels -------------

    R = len(tau) # number of annotators
    train_Y_noisy = np.zeros((train_num, R))
    transition = np.zeros((train_num, R, num_classes))

    for r in range(R):

        P = np.zeros((train_num, num_classes))

        # Sample instance flip rates: q_i for i=1,...,train_num
        flip_distribution = stats.truncnorm((0 - tau[r]) / norm_std, (0.6 - tau[r]) / norm_std, loc=tau[r], scale=norm_std)
        flip_rate = flip_distribution.rvs(train_num)

        # Independently sample w from the standard normal distribution
        W = np.random.randn(num_classes, feature_dim, num_classes) 

        for i in range(train_num):
            x = train_X[i]
            y = int(train_Y_clean[i])

            A = np.matmul(x, W[y]) # generate instance-dependent flip rates 
            A[y] = -inf # control the diagonal entry of the instance-dependent transition matrix
            A = flip_rate[i] * softmax(A, axis=0) # make the sum of the off-diagonal entries of the y_i-th row to be q_i
            A[y] += 1 - flip_rate[i] # set the diagonal entry to be 1-q_i

            P[i] = A.cpu().numpy()
        
        transition[:, r, :] = P
        new_label = [np.random.choice([0, 1], p=P[i]) for i in range(train_num)]
        train_Y_noisy[:, r] = new_label

    # ------------- returned dataset: -------------

    synthetic_train = {"X": train_X,
                       "Y_clean": train_Y_clean,
                       "Y_noisy": train_Y_noisy,
                       "transition": transition}
    synthetic_test = {"X": test_X,
                      "Y": test_Y}
    
    return synthetic_train, synthetic_test





############################################################################################
#                                                                                          #
#                 instance-dependent noisy label: Xia et al, 2020 Nips                     #
#                                                                                          #
############################################################################################
    
def get_instance_noisy_label(dataset, labels, num_classes, feature_size, tau, norm_std=0.1, seed=1): 
    """

    Input:
        dataset -- mnist, cifar10 # not train_loader
        labels -- true labels (targets)
        label_num -- class number
        feature_size -- the size of input images (e.g. 28*28)
        tau -- list of noise rate, e.g. [0.1, 0.3, 0.5]
        norm_std -- default 0.1
        seed -- random_seed 

    Output:
        new label -- N * R
        transition matrices -- N * R * K

    """

    
    print("building dataset...")

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    label_num = num_classes
    N = labels.shape[0]
    R = len(tau) # number of annotators
    noisy_label = np.zeros((N, R))
    transition = np.zeros((N, R, num_classes))


    if isinstance(labels, list):
            labels = torch.FloatTensor(labels)
    labels = labels.to(device)

    # Sample instance flip rates: q_i for i=1,...,train_num
    flip_rate = np.zeros((R, N))
    for r in range(R):
        flip_distribution = stats.truncnorm((0 - tau[r]) / norm_std, (0.6 - tau[r]) / norm_std, loc=tau[r], scale=norm_std)
        flip_rate[r] = flip_distribution.rvs(N)
    
    # Independently sample w from the standard normal distribution
    W = np.random.randn(R, label_num, feature_size, label_num)
    W = torch.FloatTensor(W).to(device)

    # generate the instance-dependent transition matrix
    for i, (x, y) in enumerate(dataset):
        # 1*m *  m*10 = 1*10
        x = x.to(device)

        for r in range(R):
            A = x.view(1, -1).mm(W[r, y, :, :].view(feature_size, -1)).squeeze(0)
            A[y] = -inf
            A = flip_rate[r, i] * F.softmax(A, dim=0)
            A[y] += 1 - flip_rate[r, i]

            transition[i, r, :] = A.cpu().numpy()
            transition[i, r, :] /= sum(transition[i, r, :])


    # generate noisy labels
    l = [i for i in range(label_num)]
    for r in range(R):
        new_label = [np.random.choice(l, p=transition[i, r, :]) for i in range(N)]
        noisy_label[:, r] = np.array(new_label)
    

    return noisy_label, transition
    
    


def incomplete_labeling(noisy_label, R, seed, annot_num):

    print("incomplete labeling...")

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    N = noisy_label.shape[0]

    # num of selected annotations: annot_num
    rng = np.random.default_rng()

    returned_annot = -1 * np.ones((N, R))
    for n in range(N):
        selected_idx = rng.choice(np.array(range(R)), annot_num, replace=False)
        returned_annot[n, selected_idx] = noisy_label[n, selected_idx] # (N, R)

    returned_noisy_label = majority_voting(returned_annot, seed)
    
    print("Generate instance-dependent label noise: Done!")
    print("Label noise shape:", returned_annot.shape)


    return returned_annot, returned_noisy_label




def majority_voting(annot, seed=1):

    """
    computes majority voting label; ties are broken uniformly at random
        * annot: (n, R)
    """

    np.random.seed(seed)
    n = annot.shape[0]
    pred_mv = np.zeros(n)

    for i in range(n):
        annot_temp = annot[i].astype(float)
        annot_temp[annot_temp == -1] = np.nan
        modes = multimode(np.array([annot_temp, annot_temp]).reshape(-1))
        np.random.shuffle(modes)
        pred_mv[i] = modes[0]
    
    return pred_mv



############################################################################################
#                                                                                          #
#                                 Get real annotator labels                                #
#                                                                                          #
############################################################################################

def get_real_annotator_labels(annotations, K):
    """
    Input:
        annotations --- n * R (n: sample size; R: num of annotators)
        K --- num of classes

    Output:
        f --- one hot annotations, N * R * K
        answers_bin_missings --- annotations_list, N * R * K, 结果好像还是one-hot (MaxMIG)
        annotator_label --- R-element dict, annotator_softmax_label (MBEM)
                            the r-th element "softmaxr_label": N * K (和annotator的index无关？)
        annotators_per_sample --- N-element list
                                  the n-th element: idx of annotators for the n-th data point (MBEM)
        annotator_label_mask --- annotator_mask, N * R
                                 the n-th row: (n,r)=1 if the r-th annotator labels it; otherwise, (n,r)=0
    """

    N = annotations.shape[0]
    R = annotations.shape[1]

    f = np.zeros((N, R, K))
    annotator_label_mask = np.zeros((N, R))
    annotator_label = {}
    for r in range(R):
        annotator_label['softmax' + str(r) + '_label'] = np.zeros((N, K))  
    annotators_per_sample = []

    for n in range(N):
        a = np.argwhere(annotations[n,:]!= -1) 
        annotators_per_sample.append(a[:,0])  # idx of annotators for the nth data point
                                              # subvector of [0, 1, ..., R-1]
        count = 0
        for r in annotators_per_sample[n]:
            f[n, r, annotations[n, r]] = 1
            annotator_label_mask[n, r] = 1
            annotator_label['softmax' + str(count) + '_label'][n] = f[n, r, :]
            count = count + 1 
            
    answers_bin_missings = []
    #answers_bin_missings = np.zeros((N, R))
    for n in range(N):
        row = []
        for r in range(R):
            if annotations[n, r] == -1:
                row.append(0 * np.ones(K))
            else:
                row.append(one_hot(annotations[n, r], K)[0, :])
        answers_bin_missings.append(row)
    answers_bin_missings = np.array(answers_bin_missings)
    
    return f, answers_bin_missings, annotator_label, annotators_per_sample, annotator_label_mask




############################################################################################
#                                                                                          #
#                                    Tool functions                                        #
#                                                                                          #
############################################################################################

def one_hot(target, K):
    """
    K: number of classes
    target: each elment takes value in \{0, 1, ..., K-1\}
    """
    targets = np.array([target]).reshape(-1) # convert to 1-d vector
    one_hot_targets = np.eye(K)[targets] # out: n * K
    return one_hot_targets


