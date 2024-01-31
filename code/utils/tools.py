import numpy as np
import torch
import torch.nn as nn
import pdb
from numpy.random import default_rng
from statistics import multimode
from torch.nn.utils import prune

def norm(T):
    row_abs = torch.abs(T)
    row_sum = torch.sum(row_abs, 1).unsqueeze(1)
    T_norm = row_abs / row_sum
    return T_norm


def fit(X, num_classes, percentage, filter_outlier=False):
    # number of classes
    c = num_classes
    T = np.empty((c, c)) # +1 -> index 
    eta_corr = X
    ind = []
    for i in np.arange(c):
        if not filter_outlier:
            idx_best = np.argmax(eta_corr[:, i])
        else:
            eta_thresh = np.percentile(eta_corr[:, i], percentage,interpolation='higher')
            robust_eta = eta_corr[:, i]
            robust_eta[robust_eta >= eta_thresh] = 0.0
            idx_best = np.argmax(robust_eta)
            ind.append(idx_best)
        for j in np.arange(c):
            T[i, j] = eta_corr[idx_best, j]
            
    return T, ind


# train set and val set split
def dataset_split(train_images, train_labels, split_percentage=0.9, random_seed=1):
	clean_train_labels = train_labels[:, np.newaxis]
	clean_train_labels = clean_train_labels.squeeze()

	num_samples = int(clean_train_labels.shape[0])
	rng = default_rng(random_seed)
	train_set_index = rng.choice(num_samples, int(num_samples * split_percentage), replace=False)
	index = np.arange(train_images.shape[0])
	val_set_index = np.delete(index, train_set_index)

	train_set, val_set = train_images[train_set_index, :], train_images[val_set_index, :]
	train_labels, val_labels = clean_train_labels[train_set_index], clean_train_labels[val_set_index]

	return train_set, val_set, train_labels, val_labels, train_set_index, val_set_index


def one_hot(target, K):
    """
    K: number of classes
    target: each elment takes value in \{0, 1, ..., K-1\}
    """
    targets = np.array([target]).reshape(-1) # convert to 1-d vector
    one_hot_targets = np.eye(K)[targets] # out: n * K
    return one_hot_targets



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


################################## get EM labels ##################################

def DS_EM(annot, y, num_classes, tol=0.00001, max_iter=10, seed=1, majority_vote=False, noisy_label=None):

    print("Collecting EM labels...")
    np.random.seed(seed)

    N = annot.shape[0]
    R = annot.shape[1]
    K = num_classes

    # initialize
    iter = 0
    converged = False
    old_class_marginals = None
    old_error_rates = None

    if majority_vote:
        y_hat = majority_voting(annot, seed=seed) # (N, )
    else:
        y_hat = noisy_label

    y_probs = np.eye(K)[np.array(y_hat).astype(int).reshape(-1)] # (N, K) -- y_probs initialization: one hot of majority voting

    # annot -> one hot annot
    annot_one_hot = np.zeros((N * R, K))
    annot = np.array(annot).astype(int).reshape(-1) # (N * R, )
    mask = (annot != -1)
    annot_one_hot[mask] = np.eye(K)[annot[mask]]
    annot_one_hot = annot_one_hot.reshape(N, R, K)
    
    # EM

    while not converged:
        iter += 1

        # M step
        (class_marginals, error_rates) = M_step(annot_one_hot, y_probs)

        # E step
        y_probs = E_step(annot_one_hot, class_marginals, error_rates)

        # check for convergence
        if old_class_marginals is not None:

            class_marginals_diff = np.sum(np.abs(class_marginals - old_class_marginals))
            error_rates_diff = np.sum(np.abs(error_rates - old_error_rates))

            if (class_marginals_diff < tol and error_rates_diff < tol) or iter > max_iter:
                converged = True
        else:
            None
        
        # update current values
        old_class_marginals = class_marginals
        old_error_rates = error_rates
    
    preds = np.argmax(y_probs, axis=1)
    u = (preds == y).sum()
    print("Final train accuracy : %f" % (u/len(y)))

    return preds



def E_step(annot_one_hot, class_marginals, error_rates):
    """
    Use equation (2.5) and the estimates of the p's and pi's to calculate new estimates of the T's.
        * T_{ij}: data i, class j (indicator)
    """
    N = annot_one_hot.shape[0]
    R = annot_one_hot.shape[1]
    K = annot_one_hot.shape[2]

    y_probs = np.zeros((N, K))

    for i in range(N):
        for j in range(K):
            y_probs[i, j] = class_marginals[j] * np.prod(np.power(error_rates[:, j, :], annot_one_hot[i, :, :]))
        
        # normalize: summing over all classes
        sum_temp = np.sum(y_probs[i, :])
        if sum_temp > 0:
            y_probs[i, :] /= float(sum_temp)
    
    return y_probs


def M_step(annot_one_hot, y_probs):
    """
    Use equations (2.3) and (2.4) to obtain estimates of the p's and pi's.
        * p: class marginals -- estimates for the prior class probabilities (K, )
        * pi: error rates (R, K, K)
    """
    N = annot_one_hot.shape[0]
    R = annot_one_hot.shape[1]
    K = annot_one_hot.shape[2]

    # class_marginals: (p_1,...,p_K)
    class_marginals = np.sum(y_probs, 0) / float(N) # y_hat: (N, K)

    # error rates: (pi^{(r)}_{j, l}) -- the r-th annotatot; true label j; noisy annotation l
    error_rates = np.zeros((R, K, K))

    for r in range(R): # annotator: r
        for j in range(K): # true label: j
            for l in range(K): # noisy label: l
                error_rates[r, j, l] = np.dot(y_probs[:, j], annot_one_hot[:, r, l])
            
            # normalize: summing over all obervation classes
            sum_temp = np.sum(error_rates[r, j, :])
            if sum_temp > 0:
                error_rates[r, j, :] /= float(sum_temp)
    
    return (class_marginals, error_rates)


#################################### prune ####################################

# def posterior_prune(model, threshold):
    
#     # mask == 0: cut

#     for name, module in model.named_children():

#         cond = True

#         if name[:4] == 'out_':
#             cond = False
#         elif name[:4] == 'conv':
#             cond = False

#         if cond:

#             mask_weight = torch.abs(module.state_dict()["weight"]) > threshold  # > threshold: keep the param
#             prune.custom_from_mask(module=module, name='weight', mask=mask_weight)

#             mask_bias = torch.abs(module.state_dict()["bias"]) > threshold  # > threshold: keep the param
#             prune.custom_from_mask(module=module, name='bias', mask=mask_bias)

def posterior_prune(model, threshold):
    
    # mask == 0: cut

    for name, module in model.named_modules():

        if name[:4] == 'out_':
            continue
        
        elif (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)):

            if module.weight is not None:
                mask_weight = torch.abs(module.weight) > threshold  # > threshold: keep the param
                prune.custom_from_mask(module=module, name='weight', mask=mask_weight)
            
            if module.bias is not None:
                mask_bias = torch.abs(module.bias) > threshold  # > threshold: keep the param
                prune.custom_from_mask(module=module, name='bias', mask=mask_bias)








