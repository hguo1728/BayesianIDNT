import os

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector
from torch.autograd import Variable

import argparse
import logging

import numpy as np
from numpy.random import default_rng
import random
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt

from utils.synthetic import *
from utils.tools import *
from Data_load.trusted import *
from Data_load.mnist import *
from Data_load.cifar10 import *
from Data_load.cifar100 import *
from Data_load.labelme import *
from Data_load.transformer import *
from models import LeNet, ResNet, VGG, Transition, FCNN



########################################################################################
#                                                                                      #
#                              Train: proposed methods                                 #
#                                                                                      #
########################################################################################


def train_ours(args,alg_options,logger, t):

    logger = logging.getLogger()

    train_loader_no_shuffle = alg_options['train_loader_no_shuffle']
    train_loader = alg_options['train_loader']
    val_loader = alg_options['val_loader']
    test_loader = alg_options["test_loader"]

    Num_train = alg_options['train_dataset'].__len__()
    Num_val = alg_options['val_dataset'].__len__()
    Num_classes = args.num_classes
    Num_annotator = args.R

    device = alg_options['device']

    # models

    if args.dataset == "synthetic":
        pass

    elif args.dataset == "mnist":

        model_1 = LeNet.Lenet()
        model_1 = model_1.to(device)

        model_2 = LeNet.Lenet()
        model_2 = model_2.to(device)

        model_T = LeNet.Lenet_T(Num_annotator, Num_classes)
        model_T = model_T.to(device)

    elif args.dataset == "cifar10":

        model_1 = ResNet.ResNet18(Num_classes)
        model_1 = model_1.to(device)

        model_2 = ResNet.ResNet18(Num_classes)
        model_2 = model_2.to(device)

        model_T = ResNet.ResNet18_T(num_classes=Num_classes, R=Num_annotator)
        model_T = model_T.to(device)

    elif args.dataset == "cifar100":

        model_1 = ResNet.ResNet34(Num_classes)
        model_1 = model_1.to(device)

        model_2 = ResNet.ResNet34(Num_classes)
        model_2 = model_2.to(device)

        model_T = ResNet.ResNet34_T(num_classes=Num_classes, R=Num_annotator)
        model_T = model_T.to(device)

    elif args.dataset == "labelme":
        model_1 = FCNN.FCNN()
        model_1 = model_1.to(device)

        model_2 = FCNN.FCNN()
        model_2 = model_2.to(device)

        model_T = FCNN.FCNN_T()
        model_T = model_T.to(device)

    else:
        logger.info('Incorrect choice for dataset')
    
    
    error_rate = args.error_rate
    linear_LRT = args.n_epoch - args.n_epoch_burn

    # LRT ratios: collect NEW trusted labels
    
    LRT_ratio_start = (1 - error_rate) * args.start_LRT
    LRT_ratio_end = (1 - error_rate) * args.end_LRT

    LRT_ratios = np.ones(linear_LRT) * LRT_ratio_end
    LRT_ratios[: linear_LRT] = np.linspace(LRT_ratio_start, LRT_ratio_end, linear_LRT)

    # LRT ratios: DELETE old trusted labels

    LRT_ratio_start_2 = args.thr_Omega_trusted
    LRT_ratio_end_2 = (1 - error_rate) * args.thr_Omega_trusted_end

    LRT_ratios_2 = np.ones(linear_LRT) * LRT_ratio_end_2
    LRT_ratios_2[: linear_LRT] = np.linspace(LRT_ratio_start_2, LRT_ratio_end_2, linear_LRT)


    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    scheduler_1 = optim.lr_scheduler.OneCycleLR(optimizer_1, args.learning_rate, epochs=args.n_epoch, steps_per_epoch=len(train_loader), verbose=True)

    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    scheduler_2 = optim.lr_scheduler.OneCycleLR(optimizer_2, args.learning_rate, epochs=args.n_epoch, steps_per_epoch=len(train_loader), verbose=True)

    optimizer_T = torch.optim.Adam(model_T.parameters(), lr=args.learning_rate_T)
    scheduler_T = optim.lr_scheduler.OneCycleLR(optimizer_T, args.learning_rate_T, epochs=args.n_epoch_T_init+args.n_epoch_T_fine_tune, steps_per_epoch=len(train_loader), verbose=True)

    # base - 1
    Returned_Train_ACC_1 = []
    Returned_Val_ACC_1 = []
    Returned_Test_ACC_1 = []
    Returned_Train_LOSS_1 = []
    Returned_Val_LOSS_1 = []

    Returned_LRT_Train_data_num_1 = []
    Returned_LRT_Val_data_num_1 = []
    Returned_LRT_Train_data_ACC_1 = []
    Returned_LRT_Val_data_ACC_1 = []

    # base - 2
    Returned_Train_ACC_2 = []
    Returned_Val_ACC_2 = []
    Returned_Test_ACC_2 = []
    Returned_Train_LOSS_2 = []
    Returned_Val_LOSS_2 = []

    Returned_LRT_Train_data_num_2 = []
    Returned_LRT_Val_data_num_2 = []
    Returned_LRT_Train_data_ACC_2 = []
    Returned_LRT_Val_data_ACC_2 = []


    # transition
    Returned_Train_ESTIMATION_ERROR = []
    Returned_Val_ESTIMATION_ERROR = []


    ############################## Warm up the classifier; select trusted sample ##############################

    logger.info("################ warm up the classifier ################")

    best_loss_1 = inf
    best_loss_2 = inf
    best_val_acc_1 = 0
    best_val_acc_2 = 0
    test_acc_on_best_model_1 = 0
    test_acc_on_best_model_2 = 0

    for epoch in range(args.n_epoch_burn):

        logger.info(" ")
        logger.info("-------- Epoch %d --------", epoch+1)

        # train
        logger.info("Training......")
        model_1.train()
        model_2.train()
        train_acc_1, train_acc_2, train_loss_1, train_loss_2 = train_warm(train_loader, epoch, model_1, optimizer_1, model_2, optimizer_2, scheduler_1, scheduler_2, alg_options, args)

        logger.info("Train ACC_1: {:.4f}".format(train_acc_1))
        logger.info("Train ACC_2: {:.4f}".format(train_acc_2))
        logger.info("Train loss_1: {:.4f}".format(train_loss_1))
        logger.info("Train loss_2: {:.4f}".format(train_loss_2))

        Returned_Train_ACC_1.append(train_acc_1)
        Returned_Train_ACC_2.append(train_acc_2)
        Returned_Train_LOSS_1.append(train_loss_1)
        Returned_Train_LOSS_2.append(train_loss_2)

        # Validation
        logger.info("Validation......")

        if args.dataset != "labelme":
            val_acc_1, val_acc_2, val_loss_1, val_loss_2 = eval_warm(val_loader, model_1, model_2, device)
            
            logger.info("Validation ACC_1: {:.4f}".format(val_acc_1))
            logger.info("Validation ACC_1: {:.4f}".format(val_acc_2))
            logger.info("Validation loss_1: {:.4f}".format(val_loss_1))
            logger.info("Validation loss_2: {:.4f}".format(val_loss_2))

            Returned_Val_ACC_1.append(val_acc_1)
            Returned_Val_ACC_2.append(val_acc_2)
            Returned_Val_LOSS_1.append(val_loss_1)
            Returned_Val_LOSS_2.append(val_loss_2)
        
        else:
            val_acc_1, val_acc_2 = evaluate(val_loader, model_1, model_2, device)
            Returned_Val_ACC_1.append(val_acc_1)
            Returned_Val_ACC_2.append(val_acc_2)

        # Test ACC
        test_acc_1, test_acc_2 = evaluate(test_loader, model_1, model_2, device)

        logger.info("Test......")
        logger.info("Model 1: Test ACC: {:.4f}".format(test_acc_1))
        logger.info("Model 2: Test ACC: {:.4f}".format(test_acc_2))

        Returned_Test_ACC_1.append(test_acc_1)
        Returned_Test_ACC_2.append(test_acc_2)

        if args.dataset != "labelme":

            if val_loss_1 < best_loss_1:
                best_loss_1 = val_loss_1
                test_acc_on_best_model_1 = test_acc_1
                torch.save(model_1.state_dict(), args.log_folder + '/trial_' + str(t) + '/model_1.pth')
                print('Warmup: Model 1 Saved')
            
            if val_loss_2 < best_loss_2:
                best_loss_2 = val_loss_2
                test_acc_on_best_model_2 = test_acc_2
                torch.save(model_2.state_dict(), args.log_folder + '/trial_' + str(t) + '/model_2.pth')
                print('Warmup: Model 2 Saved')
        
        else:

            if val_acc_1 > best_val_acc_1:
                best_val_acc_1 = val_acc_1
                test_acc_on_best_model_1 = test_acc_1
                torch.save(model_1.state_dict(), args.log_folder + '/trial_' + str(t) + '/model_1.pth')
                print('Warmup: Model 1 Saved')
            
            if val_acc_2 > best_val_acc_2:
                best_val_acc_2 = val_acc_2
                test_acc_on_best_model_2 = test_acc_2
                torch.save(model_2.state_dict(), args.log_folder + '/trial_' + str(t) + '/model_2.pth')
                print('Warmup: Model 2 Saved')
        
        print("Best model_1: ACC(test) {}; Loss(val) {}".format(test_acc_on_best_model_1, best_loss_1))
        print("Best model_2: ACC(test) {}; Loss(val) {}".format(test_acc_on_best_model_2, best_loss_2))
        
    logger.info("Test ACC on the chosen model_1:{:.4f}".format(test_acc_on_best_model_1))
    logger.info("Test ACC on the chosen model_2:{:.4f}".format(test_acc_on_best_model_2))





    logger.info("################ collect trusted labels (init) ################")

    logger.info("*---*---*---*---* Collect trusted examples (init) *---*---*---*---*")

    model_1.load_state_dict(torch.load(args.log_folder + '/trial_' + str(t) + '/model_1.pth'))
    model_2.load_state_dict(torch.load(args.log_folder + '/trial_' + str(t) + '/model_2.pth'))

    model_1.eval()
    model_2.eval()

    threshold = args.thr

    # ----------- Train -----------

    trusted_idx_1, trusted_idx_2, trusted_labels_1, trusted_labels_2 = get_trusted(model_1, model_2, train_loader_no_shuffle, threshold, device)

    # model 1
    trusted_labels_true_1 = alg_options['train_dataset'].train_labels[trusted_idx_1]

    Train_data_num_1 = len(trusted_idx_1)
    Train_data_ACC_1 = (np.array(trusted_labels_1) == np.array(trusted_labels_true_1)).sum() / Train_data_num_1

    Returned_LRT_Train_data_num_1.append(Train_data_num_1)
    Returned_LRT_Train_data_ACC_1.append(Train_data_ACC_1)

    logger.info("Model 1 (train) -- Selected trusted examples ACC: {} (Num: {})".format(Train_data_ACC_1, Train_data_num_1))
    alg_options['train_dataset'].update_trusted_label_1(trusted_labels_1, trusted_idx_1)


    # model 2
    trusted_labels_true_2 = alg_options['train_dataset'].train_labels[trusted_idx_2]

    Train_data_num_2 = len(trusted_idx_2)
    Train_data_ACC_2 = (np.array(trusted_labels_2) == np.array(trusted_labels_true_2)).sum() / Train_data_num_2

    Returned_LRT_Train_data_num_2.append(Train_data_num_2)
    Returned_LRT_Train_data_ACC_2.append(Train_data_ACC_2)

    logger.info("Model 2 (train) -- Selected trusted examples ACC: {} (Num: {})".format(Train_data_ACC_2, Train_data_num_2))
    alg_options['train_dataset'].update_trusted_label_2(trusted_labels_2, trusted_idx_2)


    # ----------- Validation -----------

    if args.dataset != "labelme":

        trusted_idx_1, trusted_idx_2, trusted_labels_1, trusted_labels_2 = get_trusted(model_1, model_2, val_loader, threshold, device)

        # model 1
        trusted_labels_true_1 = alg_options['val_dataset'].val_labels[trusted_idx_1]

        Val_data_num_1 = len(trusted_idx_1)
        Val_data_ACC_1 = (np.array(trusted_labels_1) == np.array(trusted_labels_true_1)).sum() / Val_data_num_1

        Returned_LRT_Val_data_num_1.append(Val_data_num_1)
        Returned_LRT_Val_data_ACC_1.append(Val_data_ACC_1)

        logger.info("Model 1 (val) -- Selected trusted examples ACC: {} (Num: {})".format(Val_data_ACC_1, Val_data_num_1))
        alg_options['val_dataset'].update_trusted_label_1(trusted_labels_1, trusted_idx_1)

        # model 2
        trusted_labels_true_2 = alg_options['val_dataset'].val_labels[trusted_idx_2]

        Val_data_num_2 = len(trusted_idx_2)
        Val_data_ACC_2 = (np.array(trusted_labels_2) == np.array(trusted_labels_true_2)).sum() / Val_data_num_2

        Returned_LRT_Val_data_num_2.append(Val_data_num_2)
        Returned_LRT_Val_data_ACC_2.append(Val_data_ACC_2)

        logger.info("Model 2 (val) -- Selected trusted examples ACC: {} (Num: {})".format(Val_data_ACC_2, Val_data_num_2))
        alg_options['val_dataset'].update_trusted_label_2(trusted_labels_2, trusted_idx_2)

   

    logger.info("*---*---*---*---* Get init error rates (instance-independent) *---*---*---*---*")

    # annot -> one hot annot 

    # train
    annot_train = alg_options['train_dataset'].get_annot(idx=range(Num_train)) # (n, R)
    annot_one_hot = torch.zeros((Num_train * Num_annotator, Num_classes)).to(device)
    annot_train = annot_train.reshape(-1).to(device)
    mask = (annot_train != -1)
    annot_one_hot[mask] = F.one_hot(annot_train[mask].long(), Num_classes).float()
    annot_one_hot = annot_one_hot.reshape(Num_train, Num_annotator, Num_classes) # (n, R, K)

    # val

    if args.dataset != "labelme":
   
        annot_one_hot_val = torch.zeros((Num_val * Num_annotator, Num_classes)).to(device)
        annot_val = alg_options['val_dataset'].get_annot(idx=range(Num_val))
        annot_val = annot_val.reshape(-1).to(device)
        mask = (annot_val != -1)
        annot_one_hot_val[mask] = F.one_hot(annot_val[mask].long(), Num_classes).float()
        annot_one_hot_val = annot_one_hot_val.reshape(Num_val, Num_annotator, Num_classes)

        annot_one_hot_cat = torch.cat((annot_one_hot, annot_one_hot_val), 0)
    
    else:
        annot_one_hot_cat = annot_one_hot
    
    annot_one_hot_cat = annot_one_hot_cat.detach().cpu().numpy()
    


    # trusted label -> one hot trusted

    # model 1: train
    trusted_labels_init_1 = alg_options['train_dataset'].get_trusted_label_1(idx=range(Num_train)).to(device) # (n, )
    trusted_init_one_hot_1 = torch.zeros((Num_train, Num_classes)).to(device) # (n, K)
    mask = (trusted_labels_init_1 != -1)
    trusted_init_one_hot_1[mask] = F.one_hot(trusted_labels_init_1[mask].long(), Num_classes).float()

    # model 1: val
    if args.dataset != "labelme":
        trusted_labels_init_val_1 = alg_options['val_dataset'].get_trusted_label_1(idx=range(Num_val)).to(device) # (n, )
        trusted_init_one_hot_val_1 = torch.zeros((Num_val, Num_classes)).to(device) # (n, K)
        mask = (trusted_labels_init_val_1 != -1)
        trusted_init_one_hot_val_1[mask] = F.one_hot(trusted_labels_init_val_1[mask].long(), Num_classes).float()

        trusted_init_one_hot_cat_1 = torch.cat((trusted_init_one_hot_1, trusted_init_one_hot_val_1), 0)
    
    else:
        trusted_init_one_hot_cat_1 = trusted_init_one_hot_1

    # model 2: train
    trusted_labels_init_2 = alg_options['train_dataset'].get_trusted_label_2(idx=range(Num_train)).to(device) # (n, )
    trusted_init_one_hot_2 = torch.zeros((Num_train, Num_classes)).to(device) # (n, K)
    mask = (trusted_labels_init_2 != -1)
    trusted_init_one_hot_2[mask] = F.one_hot(trusted_labels_init_2[mask].long(), Num_classes).float()

    # model 2: val
    if args.dataset != "labelme":
        trusted_labels_init_val_2 = alg_options['val_dataset'].get_trusted_label_2(idx=range(Num_val)).to(device) # (n, )
        trusted_init_one_hot_val_2 = torch.zeros((Num_val, Num_classes)).to(device) # (n, K)
        mask = (trusted_labels_init_val_2 != -1)
        trusted_init_one_hot_val_2[mask] = F.one_hot(trusted_labels_init_val_2[mask].long(), Num_classes).float()

        trusted_init_one_hot_cat_2 = torch.cat((trusted_init_one_hot_2, trusted_init_one_hot_val_2), 0)
    
    else:
        trusted_init_one_hot_cat_2 = trusted_init_one_hot_2



    # error rates 1 (init: instance-independent) -- model 1
    est_error_rates_1 = get_error_rates(annot_one_hot_cat, trusted_init_one_hot_cat_1.detach().cpu().numpy())
    est_error_rates_1 = torch.from_numpy(est_error_rates_1).to(device)

    # error rates 2 (init: instance-independent) -- model 2
    est_error_rates_2 = get_error_rates(annot_one_hot_cat, trusted_init_one_hot_cat_2.detach().cpu().numpy())
    est_error_rates_2 = torch.from_numpy(est_error_rates_2).to(device)

    # error rates
    est_error_rates = (est_error_rates_1 + est_error_rates_2) / 2 


    ############################## Transition model: initial training ##############################

    logger.info("################ start to train the transition model (init) ################")

    
    # ----------- transition matrix (init): train ----------- 
    logger.info("*---*---*---*---* transition matrix (init): train *---*---*---*---*")

    best_loss = inf
    val_estimation_error_on_best_model = 0
    train_estimation_error_on_best_model = 0

    model_T.get_pretrained_weights(model_2)

    for epoch in range(args.n_epoch_T_init):
        logger.info("-------- Epoch %d --------", epoch+1)

        # MAP......
        logger.info("MAP Training......")
        model_T.train()

        if args.annotator_type != 'real':
            train_loss, train_estimation_error = train_T_MAP(model_T, train_loader, optimizer_T, scheduler_T, alg_options, args, epoch, est_error_rates)
            logger.info("Train loss: {:.4f}".format(train_loss))
            logger.info("Estimation error of transition matrix (train): {}".format(train_estimation_error))
        else:
            train_loss = train_T_MAP(model_T, train_loader, optimizer_T, scheduler_T, alg_options, args, epoch, est_error_rates)
            logger.info("Train loss: {:.4f}".format(train_loss))

        # Evaluation......
        logger.info("Evaluation......")

        # Tr
        if args.annotator_type != 'real':
            tr_loss, tr_estimation_error = eval_T(train_loader, model_T, alg_options, args, est_error_rates, train=True)
            logger.info("Tr loss: {:.4f}".format(tr_loss))
            logger.info("Tr estimation error: {}".format(tr_estimation_error))
            Returned_Train_ESTIMATION_ERROR.append(tr_estimation_error)
        else:
            tr_loss = eval_T(train_loader, model_T, alg_options, args, est_error_rates, train=True)
            logger.info("Tr loss: {:.4f}".format(tr_loss))
    

        # Val
        if args.dataset != "labelme":
            if args.annotator_type != 'real':
                val_loss, val_estimation_error = eval_T(val_loader, model_T, alg_options, args, est_error_rates)
                logger.info("Val loss: {:.4f}".format(val_loss))
                logger.info("Val estimation error: {}".format(val_estimation_error))
                Returned_Val_ESTIMATION_ERROR.append(val_estimation_error)
            else:
                val_loss = eval_T(val_loader, model_T, alg_options, args, est_error_rates)
                logger.info("Val loss: {:.4f}".format(val_loss))
            

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model_T.state_dict(), args.log_folder + '/trial_' + str(t) + '/transition_model.pth')

                if args.annotator_type != 'real':
                    train_estimation_error_on_best_model = tr_estimation_error
                    val_estimation_error_on_best_model = val_estimation_error
                    print("Best model estimation error (train): {}".format(train_estimation_error_on_best_model))
                    print("Best model estimation error (val): {}".format(val_estimation_error_on_best_model))

            
    
    # ----------- transition matrix (init): fine tune ----------- 
    logger.info("*---*---*---*---* transition matrix (init): fine tune *---*---*---*---*")

    best_loss = inf
    val_estimation_error_on_best_model = 0
    train_estimation_error_on_best_model = 0

    lambda_n = args.lambdan
    sigma1= args.sigma1
    sigma0= args.sigma0

    try:
        model_T.load_state_dict(torch.load(args.log_folder + '/trial_' + str(t) + '/transition_model.pth'))
    except:
        torch.save(model_T.state_dict(), args.log_folder + '/trial_' + str(t) + '/transition_model.pth')

    model_T.eval()

    sparse_threshold = np.sqrt(np.log((1 - lambda_n) / lambda_n * np.sqrt(sigma1 / sigma0)) / (0.5 / sigma0 - 0.5 / sigma1))
    posterior_prune(model_T, sparse_threshold)

    # fine tune

    for epoch in range(args.n_epoch_T_fine_tune):
        logger.info("-------- Init Fine Tune: Epoch %d --------", epoch+1)

        # MAP......
        logger.info("MAP Training (fine tune)......")
        model_T.train()

        if args.annotator_type != 'real':
            loss_fine_tune, estimation_error_fine_tune = train_T_MAP(model_T,train_loader, optimizer_T, scheduler_T, alg_options, args, epoch, est_error_rates, masked=True)
            logger.info("Train loss (fine tune): {:.4f}".format(loss_fine_tune))
            logger.info("Estimation error of transition matrix (fine tune): {}".format(estimation_error_fine_tune))
        else:
            loss_fine_tune = train_T_MAP(model_T,train_loader, optimizer_T, scheduler_T, alg_options, args, epoch, est_error_rates, masked=True)
            logger.info("Train loss (fine tune): {:.4f}".format(loss_fine_tune))


        # Evaluation......
        logger.info("Evaluation (fine tuned)......")
        
        # Tr
        if args.annotator_type != 'real':
            tr_loss_tuned, tr_estimation_error_tuned = eval_T(train_loader, model_T, alg_options, args, est_error_rates, masked=True, train=True)
            logger.info("Tr loss (fine tuned): {:.4f}".format(tr_loss_tuned))
            logger.info("Tr estimation error (fine tuned): {}".format(tr_estimation_error_tuned))
            Returned_Train_ESTIMATION_ERROR.append(tr_estimation_error_tuned)
        else:
            tr_loss_tuned = eval_T(train_loader, model_T, alg_options, args, est_error_rates, masked=True, train=True)
            logger.info("Tr loss (fine tuned): {:.4f}".format(tr_loss_tuned))
            

        # Validation
        if args.dataset != "labelme":
            if args.annotator_type != 'real':
                val_loss_tuned, val_estimation_error_tuned = eval_T(val_loader, model_T, alg_options, args, est_error_rates, masked=True)
                logger.info("Validation loss (fine tuned): {:.4f}".format(val_loss_tuned))
                logger.info("Estimation error of transition matrix (val -- fine tuned): {}".format(val_estimation_error_tuned))
            else:
                val_loss_tuned = eval_T(val_loader, model_T, alg_options, args, est_error_rates, masked=True)
                logger.info("Validation loss (fine tuned): {:.4f}".format(val_loss_tuned))

    torch.save(model_T.state_dict(), args.log_folder + '/trial_' + str(t) + '/transition_model.pth')
        
    # sparsity: 
    total_num_para = 0
    non_zero_element = 0
    for name, param in model_T.named_parameters():
        total_num_para += param.numel()
        non_zero_element += (param.abs() > 1e-6).sum()
    sparsity_rate = non_zero_element.item() / total_num_para
    print('sparsity:', sparsity_rate)

    
    
    logger.info("################################ Train! ################################")

    best_loss_1 = inf
    best_loss_2 = inf
    best_val_acc_1 = 0
    best_val_acc_2 = 0
    test_acc_on_best_model_1 = 0
    test_acc_on_best_model_2 = 0

    for epoch in range(args.n_epoch-args.n_epoch_burn):
       
       logger.info(" ")
       logger.info("-------- [Epoch %d] --------", epoch+1+args.n_epoch_burn)


       # ------------------------- Collect new trusted data ------------------------- #
       # Pairwise LTR test

       if (args.epoch_start_LRT <= epoch) and ((epoch - args.epoch_start_LRT) % args.freq_LRT == 0):
           
           logger.info("*---*---*---*---* Pairwise LRT test: update labels *---*---*---*---*")
           
           thr_Omega_trusted = args.thr_Omega_trusted
           thr_Omega_new = args.thr_Omega_new + args.thr_increment_LRT * (epoch - args.epoch_start_LRT)
           ratio = args.trusted_ratio_max

           # -------------- train --------------

           # before
           mask = alg_options['train_dataset'].train_label_trusted_1 != -1
           
           print('Model 1: ACC_before (Train): {} (Num: {})'.format((np.array(alg_options['train_dataset'].train_labels[mask]) == alg_options['train_dataset'].train_label_trusted_1[mask]).sum() / sum(mask), sum(mask)))

           mask = alg_options['train_dataset'].train_label_trusted_2 != -1
           print('Model 2: ACC_before (Train): {} (Num: {})'.format((np.array(alg_options['train_dataset'].train_labels[mask]) == alg_options['train_dataset'].train_label_trusted_2[mask]).sum() / sum(mask), sum(mask)))

           # get trusted
           update_idx_1, update_idx_2, update_labels_1, update_labels_2 = pairwise_LRT_update(model_1, model_2, model_T, train_loader_no_shuffle, args, LRT_ratios_2[epoch], LRT_ratios[epoch], device, logger)

           alg_options['train_dataset'].update_trusted_label_1(update_labels_1, update_idx_1)
           alg_options['train_dataset'].update_trusted_label_2(update_labels_2, update_idx_2)

           # after

           # model 1
           mask = alg_options['train_dataset'].train_label_trusted_1 != -1

           Train_data_num_1 = sum(mask)
           Train_data_ACC_1 = (np.array(alg_options['train_dataset'].train_labels[mask]) == alg_options['train_dataset'].train_label_trusted_1[mask]).sum() / Train_data_num_1

           Returned_LRT_Train_data_num_1.append(Train_data_num_1)
           Returned_LRT_Train_data_ACC_1.append(Train_data_ACC_1)

           print('Model 1: ACC_after (train): {} (Num: {})'.format(Train_data_ACC_1, Train_data_num_1))

           # model 2
           mask = alg_options['train_dataset'].train_label_trusted_2 != -1

           Train_data_num_2 = sum(mask)
           Train_data_ACC_2 = (np.array(alg_options['train_dataset'].train_labels[mask]) == alg_options['train_dataset'].train_label_trusted_2[mask]).sum() / Train_data_num_2

           Returned_LRT_Train_data_num_2.append(Train_data_num_2)
           Returned_LRT_Train_data_ACC_2.append(Train_data_ACC_2)

           print('Model 2: ACC_after (train): {} (Num: {})'.format(Train_data_ACC_2, Train_data_num_2))


           # -------------- validation --------------

           if args.dataset != "labelme":

                # before
                mask = alg_options['val_dataset'].val_label_trusted_1 != -1
                print('Model 1: ACC_before (val): {} (Num: {})'.format((np.array(alg_options['val_dataset'].val_labels[mask]) == alg_options['val_dataset'].val_label_trusted_1[mask]).sum() / sum(mask), sum(mask)))

                mask = alg_options['val_dataset'].val_label_trusted_2 != -1
                print('Model 2: ACC_before (val): {} (Num: {})'.format((np.array(alg_options['val_dataset'].val_labels[mask]) == alg_options['val_dataset'].val_label_trusted_2[mask]).sum() / sum(mask), sum(mask)))

                # get trusted
                update_idx_1, update_idx_2, update_labels_1, update_labels_2 = pairwise_LRT_update(model_1, model_2, model_T, val_loader, args, LRT_ratios_2[epoch], LRT_ratios[epoch], device, logger)

                alg_options['val_dataset'].update_trusted_label_1(update_labels_1, update_idx_1)
                alg_options['val_dataset'].update_trusted_label_2(update_labels_2, update_idx_2)

                # after

                # model 1
                mask = alg_options['val_dataset'].val_label_trusted_1 != -1

                Val_data_num_1 = sum(mask)
                Val_data_ACC_1 = (np.array(alg_options['val_dataset'].val_labels[mask]) == alg_options['val_dataset'].val_label_trusted_1[mask]).sum() / Val_data_num_1

                Returned_LRT_Val_data_num_1.append(Val_data_num_1)
                Returned_LRT_Val_data_ACC_1.append(Val_data_ACC_1)

                print('Model 1: ACC_after (val): {} (Num: {})'.format(Val_data_ACC_1, Val_data_num_1))

                # model 2
                mask = alg_options['val_dataset'].val_label_trusted_2 != -1

                Val_data_num_2 = sum(mask)
                Val_data_ACC_2 = (np.array(alg_options['val_dataset'].val_labels[mask]) == alg_options['val_dataset'].val_label_trusted_2[mask]).sum() / Val_data_num_2

                Returned_LRT_Val_data_num_2.append(Val_data_num_2)
                Returned_LRT_Val_data_ACC_2.append(Val_data_ACC_2)

                print('Model 2: ACC_after (val): {} (Num: {})'.format(Val_data_ACC_2, Val_data_num_2))

  
       # ------------------------- Update the classsifier ------------------------- #

       if epoch == args.epoch_start_LRT:
           best_loss = inf

       # Train
       logger.info("Training......")
       train_acc_1, train_acc_2, train_loss_1, train_loss_2 = train_base(model_1, model_2, train_loader, epoch, optimizer_1, optimizer_2, scheduler_1, scheduler_2, alg_options, args)

       logger.info("Model 1 (Train) -- ACC: {:.4f}; Loss: {:.4f}".format(train_acc_1, train_loss_1))
       logger.info("Model 2 (Train) -- ACC: {:.4f}; Loss: {:.4f}".format(train_acc_2, train_loss_2))
       
       Returned_Train_ACC_1.append(train_acc_1)
       Returned_Train_ACC_2.append(train_acc_2)

       Returned_Train_LOSS_1.append(train_loss_1)
       Returned_Train_LOSS_2.append(train_loss_2)

       # Validation
       if args.dataset != "labelme":
            logger.info("Validation......")
            val_acc_1, val_acc_2, val_loss_1, val_loss_2 = eval_base(val_loader, model_1, model_2, alg_options, args)

            logger.info("Model 1 (Val) -- ACC: {:.4f}; Loss: {:.4f}".format(val_acc_1, val_loss_1))
            logger.info("Model 2 (Val) -- ACC: {:.4f}; Loss: {:.4f}".format(val_acc_2, val_loss_2))

            Returned_Val_ACC_1.append(val_acc_1)
            Returned_Val_ACC_2.append(val_acc_2)

            Returned_Val_LOSS_1.append(val_loss_1)
            Returned_Val_LOSS_2.append(val_loss_2)
       else:
           logger.info("Validation......")
           val_acc_1, val_acc_2 = evaluate(val_loader, model_1, model_2, device) 

           Returned_Val_ACC_1.append(val_acc_1)
           Returned_Val_ACC_2.append(val_acc_2)


       # Test ACC
       test_acc_1, test_acc_2 = evaluate(test_loader, model_1, model_2, device) 
       logger.info("Test......")
       logger.info("Model 1: Test ACC: {:.4f}".format(test_acc_1))
       logger.info("Model 2: Test ACC: {:.4f}".format(test_acc_2))

       Returned_Test_ACC_1.append(test_acc_1)
       Returned_Test_ACC_2.append(test_acc_2)

       if args.dataset != "labelme":

            if val_loss_1 < best_loss_1:
                    best_loss_1 = val_loss_1
                    test_acc_on_best_model_1 = test_acc_1
                    torch.save(model_1.state_dict(), args.log_folder + '/trial_' + str(t) + '/model_1.pth')
                    print('Training: Model 1 Saved')

            if val_loss_2 < best_loss_2:
                    best_loss_2 = val_loss_2
                    test_acc_on_best_model_2 = test_acc_2
                    torch.save(model_2.state_dict(), args.log_folder + '/trial_' + str(t) + '/model_2.pth')
                    print('Training: Model 2 Saved')

       else:
           
           if val_acc_1 > best_val_acc_1:
               best_val_acc_1 = val_acc_1
               test_acc_on_best_model_1 = test_acc_1
               torch.save(model_1.state_dict(), args.log_folder + '/trial_' + str(t) + '/model_1.pth')
               print('Training: Model 1 Saved')
               
           if val_acc_2 > best_val_acc_2:
               best_val_acc_2 = val_acc_2
               test_acc_on_best_model_2 = test_acc_2
               torch.save(model_2.state_dict(), args.log_folder + '/trial_' + str(t) + '/model_2.pth')
               print('Training: Model 2 Saved')
           

       print("Model 1: Chosen model ACC (test):", test_acc_on_best_model_1)
       print("Model 2: Chosen model ACC (test):", test_acc_on_best_model_2)

       
    
    # final result

    out = {}

    out["sparsity_rate"] = sparsity_rate

    out["Train_error"] = Returned_Train_ESTIMATION_ERROR
    out["Val_error"] = Returned_Val_ESTIMATION_ERROR



    out["Train_ACC_1"] = Returned_Train_ACC_1
    out["Val_ACC_1"] = Returned_Val_ACC_1
    out["Test_ACC_1"] = Returned_Test_ACC_1

    out["Train_loss_1"] = Returned_Train_LOSS_1
    out["Val_loss_1"] = Returned_Val_LOSS_1

    out['LRT_Train_data_num_1'] = Returned_LRT_Train_data_num_1
    out['LRT_Val_data_num_1'] = Returned_LRT_Val_data_num_1
    out['LRT_Train_data_ACC_1'] = Returned_LRT_Train_data_ACC_1
    out['LRT_Val_data_ACC_1'] = Returned_LRT_Val_data_ACC_1



    out["Train_ACC_2"] = Returned_Train_ACC_2
    out["Val_ACC_2"] = Returned_Val_ACC_2
    out["Test_ACC_2"] = Returned_Test_ACC_2

    out["Train_loss_2"] = Returned_Train_LOSS_2
    out["Val_loss_2"] = Returned_Val_LOSS_2

    out['LRT_Train_data_num_2'] = Returned_LRT_Train_data_num_2
    out['LRT_Val_data_num_2'] = Returned_LRT_Val_data_num_2
    out['LRT_Train_data_ACC_2'] = Returned_LRT_Train_data_ACC_2
    out['LRT_Val_data_ACC_2'] = Returned_LRT_Val_data_ACC_2




    np.save(args.log_folder + '/trial_' + str(t) + '/result_ours.npy', out)


    return test_acc_on_best_model_1, test_acc_on_best_model_2



########################################################################################
#                                                                                      #
#                                 Tool functions                                       #
#                                                                                      #
########################################################################################


###################################### Test: ACC ######################################

def evaluate(test_loader, model_1, model_2, device):

    correct_1 = 0
    correct_2 = 0
    total = 0
    
    with torch.no_grad():
        model_1.eval() 
        model_2.eval() 

        for batch_x, batch_y in test_loader:

            if torch.cuda.is_available:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
            
            probs_1 = model_1(batch_x)
            y_hat_1 = torch.max(probs_1, 1)[1]
            correct_1 += (y_hat_1 == batch_y).sum()

            probs_2 = model_2(batch_x)
            y_hat_2 = torch.max(probs_2, 1)[1]
            correct_2 += (y_hat_2 == batch_y).sum()

            total += batch_y.size(0)

        acc_1 = 100 * float(correct_1) / float(total)
        acc_2 = 100 * float(correct_2) / float(total)

    return acc_1, acc_2


###################################### Warm up ######################################

# train

def train_warm(train_loader, epoch, model_1, optimizer_1, model_2, optimizer_2, scheduler_1, scheduler_2, alg_options, args):

    device = alg_options['device']
    batch_size = args.batch_size
    total_batch = len(train_loader)

    model_1.train()
    model_2.train()

    train_total_loss_1 = 0
    train_total_correct_1 = 0 
    train_total_loss_2 = 0
    train_total_correct_2 = 0 

    train_num = 0

    loss_fn = nn.NLLLoss()

    for i, (indexes, batch_x, batch_y, trusted_label_1, trusted_label_2, annot, noisy_label) in enumerate(train_loader):

        train_num += batch_x.size(0)

        if torch.cuda.is_available:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_noisy_label = noisy_label.to(device)
        
        # model_1 and model_2: probs and y_hat

        probs1 = model_1(batch_x)
        y_hat_1 = torch.max(probs1, 1)[1]
        batch_correct_1 = (y_hat_1 == batch_y).sum()
        train_total_correct_1 += batch_correct_1

        probs2 = model_2(batch_x)
        y_hat_2 = torch.max(probs2, 1)[1]
        batch_correct_2 = (y_hat_2 == batch_y).sum()
        train_total_correct_2 += batch_correct_2

        # model_1 and model_2: loss and backprop

        loss_1 = loss_fn(torch.log(probs1 + 1e-10), batch_noisy_label)
        batch_loss_1 = loss_1.item()
        train_total_loss_1 += batch_loss_1 * batch_x.size(0)

        optimizer_1.zero_grad()
        loss_1.backward()
        optimizer_1.step()
        optimizer_1.zero_grad()


        loss_2 = loss_fn(torch.log(probs2 + 1e-10), batch_noisy_label)
        batch_loss_2 = loss_2.item()
        train_total_loss_2 += batch_loss_2 * batch_x.size(0)

        optimizer_2.zero_grad()
        loss_2.backward()
        optimizer_2.step()
        optimizer_2.zero_grad()


        # Print
        if (i + 1) % args.print_freq == 0:
            print('Epoch [%d], Iter [%d/%d], Training ACC_1: %.4F, Training ACC_2: %.4F, Loss_1: %.4f, Loss_2: %.4f'
                % (epoch + 1, i + 1, total_batch, batch_correct_1/batch_size, batch_correct_2/batch_size, batch_loss_1, batch_loss_2))
    
    scheduler_1.step()
    scheduler_2.step()
        
    train_acc_1 = float(train_total_correct_1) / float(train_num)
    train_acc_2 = float(train_total_correct_2) / float(train_num)
    train_loss_1 = float(train_total_loss_1) / float(train_num)
    train_loss_2 = float(train_total_loss_2) / float(train_num)

    return train_acc_1, train_acc_2, train_loss_1, train_loss_2


# validation

def eval_warm(val_loader, model_1, model_2, device):

    loss_fn = nn.NLLLoss()

    val_num = 0
    
    val_total_correct_1 = 0 
    val_total_correct_2 = 0 
    val_total_loss_1 = 0
    val_total_loss_2 = 0

    with torch.no_grad():

        model_1.eval()
        model_2.eval()

        for _, (indexes, batch_x, batch_y, trusted_label_1, trusted_label_2, annot, noisy_label) in enumerate(val_loader):

            if torch.cuda.is_available:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_noisy_label = noisy_label.to(device)
            
            # validation ACC

            probs_1 = model_1(batch_x)
            y_hat_1 = torch.max(probs_1, 1)[1]
            batch_correct_1 = (y_hat_1 == batch_y).sum()
            val_total_correct_1 += batch_correct_1

            probs_2 = model_2(batch_x)
            y_hat_2 = torch.max(probs_2, 1)[1]
            batch_correct_2 = (y_hat_2 == batch_y).sum()
            val_total_correct_2 += batch_correct_2


            # validation loss

            loss_1 = loss_fn(torch.log(probs_1 + 1e-10), batch_noisy_label)
            batch_loss_1 = loss_1.item()
            val_total_loss_1 += batch_loss_1 * batch_x.size(0)

            loss_2 = loss_fn(torch.log(probs_2 + 1e-10), batch_noisy_label)
            batch_loss_2 = loss_2.item()
            val_total_loss_2 += batch_loss_2 * batch_x.size(0)

            val_num += batch_x.size(0)
        
        val_acc_1 = float(val_total_correct_1) / float(val_num)
        val_acc_2 = float(val_total_correct_2) / float(val_num)
        val_loss_1 = float(val_total_loss_1) / float(val_num)
        val_loss_2 = float(val_total_loss_2) / float(val_num)
    
    return val_acc_1, val_acc_2, val_loss_1, val_loss_2


# Get trusted data (init)

def get_trusted(model_1, model_2, dataloader, threshold, device):
    
    model_1.eval()
    model_2.eval()

    trusted_idx_list_1 = []
    trusted_labels_list_1 = []

    trusted_idx_list_2 = []
    trusted_labels_list_2 = []

    with torch.no_grad():

        for _, (indexes, batch_x, batch_y, trusted_label_1, trusted_label_2, annot, noisy_label) in enumerate(dataloader):

            if torch.cuda.is_available:
                batch_x = batch_x.to(device)
            
            # model 1: trusted_label_1
            probs_1 = model_1(batch_x) # (batch_size, K)
            prob_max_1 = torch.max(probs_1, dim=1) # out: (max, max_indices) <-> (max_probs, labels)

            selected_mask_1 = prob_max_1[0] > float(threshold) # on device
            trusted_idx_list_1.extend(indexes[selected_mask_1.cpu()])
            trusted_labels_list_1.extend(prob_max_1[1].cpu()[selected_mask_1.cpu()])

            # model 2: trusted_label_2
            probs_2 = model_2(batch_x) # (batch_size, K)
            prob_max_2 = torch.max(probs_2, dim=1) # out: (max, max_indices) <-> (max_probs, labels)
            
            selected_mask_2 = prob_max_2[0] > float(threshold) # on device
            trusted_idx_list_2.extend(indexes[selected_mask_2.cpu()])
            trusted_labels_list_2.extend(prob_max_2[1].cpu()[selected_mask_2.cpu()])

    trusted_idx_1 = np.array(trusted_idx_list_1).astype(int)
    trusted_idx_2 = np.array(trusted_idx_list_2).astype(int)

    trusted_labels_1 = np.array(trusted_labels_list_1)
    trusted_labels_2 = np.array(trusted_labels_list_2)

    return trusted_idx_1, trusted_idx_2, trusted_labels_1, trusted_labels_2


# Get trusted data (LRT -- init)
def get_trusted_LRT(model_1, model_2, dataloader, error_rates, threshold, device, args):

    K = args.num_classes
    R = args.R
    num_batch = len(dataloader)

    model_1.eval()
    model_2.eval()

    trusted_idx_list_1 = []
    trusted_idx_list_2 = []
    trusted_labels_list_1 = []
    trusted_labels_list_2 = []

    for _, (indexes, batch_x, batch_y, trusted_label_1, trusted_label_2, annot, noisy_label) in enumerate(dataloader):

        batch_size = batch_x.shape[0]

        if torch.cuda.is_available:
            batch_x = batch_x.to(device)
            annot = annot.to(device) # (n, R)
        
        with torch.no_grad():
            probs_prior_1 = model_1(batch_x) # (n, K)
            probs_prior_2 = model_2(batch_x) # (n, K)

            batch_T = error_rates # (R, K, K)

            annot_one_hot = torch.zeros((batch_size, R, K)).to(device) # (n, R, K)
            annot_one_hot = annot_one_hot.view(batch_size * R, K) # (n * R, K)
            mask = (annot.view(-1) != -1).to(device) # (n * R, )
            annot_one_hot[mask] = F.one_hot(annot.view(-1)[mask], K).float()
            annot_one_hot = annot_one_hot.view(batch_size, R, K) # (n, R, K)
            
        probs_prior_1 = probs_prior_1.cpu().detach().numpy() # (n, K)
        probs_prior_2 = probs_prior_2.cpu().detach().numpy() # (n, K)

        batch_T = batch_T.cpu().detach().numpy() # (n, R, K, K)
        annot_one_hot = annot_one_hot.cpu().detach().numpy() # (n, R, K)
        
        LRT_criterion_1 = np.zeros((batch_size, K))
        LRT_criterion_2 = np.zeros((batch_size, K))
        for i in range(batch_size):
            for j in range(K):
                LRT_criterion_1[i, j] = probs_prior_1[i, j] * np.prod(np.power(batch_T[:, j, :], annot_one_hot[i, :, :]))
                LRT_criterion_2[i, j] = probs_prior_2[i, j] * np.prod(np.power(batch_T[:, j, :], annot_one_hot[i, :, :]))

            # normalize 
            sum_temp = np.sum(LRT_criterion_1[i, :])
            if sum_temp > 0:
                LRT_criterion_1[i, :] /= sum_temp
            
            sum_temp = np.sum(LRT_criterion_2[i, :])
            if sum_temp > 0:
                LRT_criterion_2[i, :] /= sum_temp
        
        # model 1: update trusted_label_1
        top_2_values_1, top_2_idx_1 = torch.topk(torch.from_numpy(LRT_criterion_1), k=2, dim=1) # (n, 2)
        LRT_ratios_1 = top_2_values_1[:, 0] / (top_2_values_1[:, 1] + 1e-10) # (n, )

        selected_mask = LRT_ratios_1 > float(threshold) # on device

        trusted_idx_list_1.extend(indexes[selected_mask.cpu()])
        trusted_labels_list_1.extend(top_2_idx_1[:, 0].cpu()[selected_mask.cpu()])

        # model 2: update trusted_label_2
        top_2_values_2, top_2_idx_2 = torch.topk(torch.from_numpy(LRT_criterion_2), k=2, dim=1) # (n, 2)
        LRT_ratios_2 = top_2_values_2[:, 0] / (top_2_values_2[:, 1] + 1e-10) # (n, )

        selected_mask = LRT_ratios_2 > float(threshold) # on device

        trusted_idx_list_2.extend(indexes[selected_mask.cpu()])
        trusted_labels_list_2.extend(top_2_idx_2[:, 0].cpu()[selected_mask.cpu()])

    trusted_idx_1 = np.array(trusted_idx_list_1).astype(int)
    trusted_idx_2 = np.array(trusted_idx_list_2).astype(int)

    trusted_labels_1 = np.array(trusted_labels_list_1)
    trusted_labels_2 = np.array(trusted_labels_list_2)

    return trusted_idx_1, trusted_idx_2, trusted_labels_1, trusted_labels_2


###################################### Update transition model ######################################


# estimate error rates
def get_error_rates(annot_one_hot, y_preds_one_hot):

    N = annot_one_hot.shape[0]
    R = annot_one_hot.shape[1]
    K = annot_one_hot.shape[2]

    # error rates: (pi^{(r)}_{j, l}) -- the r-th annotatot; true label j; noisy annotation l
    error_rates = np.zeros((R, K, K))

    for r in range(R): # annotator: r
        for j in range(K): # true label: j
            for l in range(K): # noisy label: l
                error_rates[r, j, l] = np.dot(y_preds_one_hot[:, j], annot_one_hot[:, r, l])
            
            # normalize: summing over all obervation classes
            sum_temp = np.sum(error_rates[r, j, :])
            if sum_temp > 0:
                error_rates[r, j, :] /= float(sum_temp)
    
    return error_rates


# Train: MAP

def train_T_MAP(model_T, train_loader, optimizer, scheduler, alg_options, args, epoch, error_rates, masked=False):
    device = alg_options['device']

    K = args.num_classes
    R = args.R
    Reg_lambda = args.Reg_lambda
    lambda_n = args.lambdan
    sigma1 = args.sigma1
    sigma0 = args.sigma0

    c1 = (lambda_n / (1 - lambda_n)) * np.sqrt(sigma0 / sigma1)
    c2 = 0.5 * (1 / sigma0 - 1 / sigma1)

    total_batch = len(train_loader)

    train_total_loss = 0
    total_estimation_error = torch.zeros((R,))
    train_num_loss = 0
    train_num_error = 0

    loss_fn = nn.NLLLoss(ignore_index=-1, reduction='mean')

    for batch_idx, (indexes, batch_x, batch_y, trusted_label_1, trusted_label_2, annot, noisy_label) in enumerate(train_loader):

        model_T.train()

        if torch.cuda.is_available:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device) # (batch_size,)
            batch_trusted_label_1 = trusted_label_1.to(device) # (batch_size,)
            batch_trusted_label_2 = trusted_label_2.to(device) # (batch_size,)
            batch_annot = annot.to(device) # (batch_size, R)
        
        mask_1 = (batch_trusted_label_1 != -1)
        mask_2 = (batch_trusted_label_2 != -1)

        batch_T = model_T(batch_x) # (n, R, K, K)

        batch_T_1 = batch_T[mask_1] 
        batch_T_2 = batch_T[mask_2] 

        batch_annot_1 = batch_annot[mask_1]
        batch_annot_2 = batch_annot[mask_2]

        batch_T_cat = torch.cat((batch_T_1, batch_T_2), 0)
        batch_annot_cat = torch.cat((batch_annot_1, batch_annot_2), 0)

        batch_size = batch_T_cat.shape[0]
        train_num_loss += batch_size
        train_num_error += batch_y.shape[0]

        # ----------- calculate cross entropy loss -----------


        # conditioned on true label
        label_one_hot = F.one_hot(batch_y, K).float() # (n, K)
        batch_T_cond_true = torch.einsum('nrkl, nk->nrl', [batch_T, label_one_hot]) # (n, R, K, K), (n, K) -> (n, R, K)


        # conditioned on trusted label
        trusted_label_one_hot_1 = F.one_hot(batch_trusted_label_1[mask_1], K).float() # (|masked|, K)
        trusted_label_one_hot_2 = F.one_hot(batch_trusted_label_2[mask_2], K).float() # (|masked|, K)
        trusted_label_one_hot_cat = torch.cat((trusted_label_one_hot_1, trusted_label_one_hot_2), 0)
        batch_T_cond = torch.einsum('nrkl, nk->nrl', [batch_T_cat, trusted_label_one_hot_cat]) # (|masked|, R, K, K) * (|masked|, K) -> (|masked|, R, K)
        
        # calculate cross entropy loss
        CE_loss = loss_fn(torch.log(batch_T_cond+1e-10).view(-1, K), batch_annot_cat.view(-1))
        reg_loss = Reg_lambda * torch.sum(torch.norm((batch_T - error_rates[None, :, :, :]).view(-1, K, K), dim=(1, 2))) # (n, R, K, K) -> (n * R, K, K) -> (n * R, )
        loss = CE_loss + reg_loss / batch_y.shape[0]

        batch_loss = loss.item() # average loss on the batch
        train_total_loss += batch_loss * batch_size # total loss on the batch

        optimizer.zero_grad()
        loss.backward()

        if masked: # mask has been implemented: gamma=1; use sigma1 in the prior loss 
            for name, param in model_T.named_parameters():
                if param.requires_grad == False:
                    continue
                else:
                    param.grad = param.grad.add(param, alpha=1/(sigma1 * batch_size))
        
        elif epoch>5: # mask has NOT been implemented: use posterior prob for gamma
            for name, param in model_T.named_parameters():

                if param.requires_grad == False:
                    continue
                elif name[:4] == "out_":
                    const = 1/(sigma1 * batch_size)
                else:
                    with torch.no_grad():
                        p0 = 1 / (c1 * torch.exp(param.pow(2) * c2) + 1) # P(gamma=0 | beta)
                    const = (p0.div(sigma0) + (1 - p0).div(sigma1)).div(batch_size)

                param.grad = param.grad.add(param * const)
        
        optimizer.step()
        optimizer.zero_grad()


        # transition matrix estimation loss
        if args.annotator_type != 'real':
            batch_T_true = alg_options['train_dataset'].get_true_transition(indexes).to(device) # (batch_size, R, K)
            batch_estimation_error = torch.max(torch.abs(batch_T_true - batch_T_cond_true), 2)[0].sum(0) # total estimation error on the batch
            total_estimation_error += batch_estimation_error.cpu()

        # Print
        if (batch_idx + 1) % args.print_freq == 0:
            if args.annotator_type != 'real':
                print('Epoch [{}], Iter [{}/{}], Loss: {}, Estimation error: {}'.format(epoch + 1, batch_idx + 1, total_batch, batch_loss, batch_estimation_error.detach().cpu().numpy() / float(batch_y.shape[0])))
            else:
                print('Epoch [{}], Iter [{}/{}], Loss: {}'.format(epoch + 1, batch_idx + 1, total_batch, batch_loss))
    
    scheduler.step()
    
    train_loss = float(train_total_loss) / float(train_num_loss)

    if args.annotator_type != 'real':
        estimation_error = (total_estimation_error).detach().numpy() / float(train_num_error)
        return train_loss, estimation_error
    else:
        return train_loss


# validation (transition)

def eval_T(val_loader, model_T, alg_options, args, error_rates, masked=False, train=False):

    device = alg_options['device']

    K = args.num_classes
    R = args.R
    Reg_lambda = args.Reg_lambda
    lambda_n = args.lambdan
    sigma1 = args.sigma1
    sigma0 = args.sigma0
    c1 = (lambda_n / (1 - lambda_n)) * np.sqrt(sigma0 / sigma1)
    c2 = 0.5 * (1 / sigma0 - 1 / sigma1)

    val_num_loss = 0
    val_num_error = 0
    val_total_loss = 0
    val_total_estimation_error = torch.zeros(R)

    loss_fn = nn.NLLLoss(ignore_index=-1, reduction='mean')


    with torch.no_grad():

        model_T.eval()

        for _, (indexes, batch_x, batch_y, trusted_label_1, trusted_label_2, annot, noisy_label) in enumerate(val_loader):

            if torch.cuda.is_available:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device) # (batch_size,)
                batch_trusted_label_1 = trusted_label_1.to(device) # (batch_size,)
                batch_trusted_label_2 = trusted_label_2.to(device) # (batch_size,)
                batch_annot = annot.to(device) # (batch_size, R)

            
            mask_1 = (batch_trusted_label_1 != -1)
            mask_2 = (batch_trusted_label_2 != -1)

            batch_T = model_T(batch_x) # (n, R, K, K)

            batch_T_1 = batch_T[mask_1] 
            batch_T_2 = batch_T[mask_2] 

            batch_annot_1 = batch_annot[mask_1]
            batch_annot_2 = batch_annot[mask_2]

            batch_T_cat = torch.cat((batch_T_1, batch_T_2), 0)
            batch_annot_cat = torch.cat((batch_annot_1, batch_annot_2), 0)

            batch_size = batch_y.shape[0]

            val_num_error += batch_y.shape[0]
            val_num_loss += batch_T_cat.shape[0]

            # ----------- calculate cross entropy loss -----------
            
            batch_T = model_T(batch_x)

            # conditioned on true label
            label_one_hot = F.one_hot(batch_y, K).float() # (n, K)
            batch_T_cond_true = torch.einsum('nrkl, nk->nrl', [batch_T, label_one_hot]) # (n, R, K, K), (n, K) -> (n, R, K)

            # conditioned on trusted label
            trusted_label_one_hot_1 = F.one_hot(batch_trusted_label_1[mask_1], K).float() # (|masked|, K)
            trusted_label_one_hot_2 = F.one_hot(batch_trusted_label_2[mask_2], K).float() # (|masked|, K)
            trusted_label_one_hot_cat = torch.cat((trusted_label_one_hot_1, trusted_label_one_hot_2), 0)
            batch_T_cond = torch.einsum('nrkl, nk->nrl', [batch_T_cat, trusted_label_one_hot_cat]) # (|masked|, R, K, K) * (|masked|, K) -> (|masked|, R, K)
            
            # calculate cross entropy loss
            CE_loss = loss_fn(torch.log(batch_T_cond+1e-10).view(-1, K), batch_annot_cat.view(-1))
            reg_loss = Reg_lambda * torch.sum(torch.norm((batch_T - error_rates[None, :, :, :]).view(-1, K, K), dim=(1, 2))) # (n, R, K, K) -> (n * R, K, K) -> (n * R, )
            loss = CE_loss + reg_loss / batch_y.shape[0]

            batch_loss = loss.item() # average loss on the batch
            val_total_loss += batch_loss * batch_T_cat.shape[0] # total loss on the batch

            # transition matrix estimation loss
            if args.annotator_type != 'real':
                if train:
                    batch_T_true = alg_options['train_dataset'].get_true_transition(indexes).to(device) # (batch_size, R, K)
                else:
                    batch_T_true = alg_options['val_dataset'].get_true_transition(indexes).to(device) # (batch_size, R, K)
            
                batch_estimation_error = torch.max(torch.abs(batch_T_true - batch_T_cond_true), 2)[0].sum(0) # total estimation error on the batch
                val_total_estimation_error += batch_estimation_error.cpu()
        
    val_loss = float(val_total_loss) / float(val_num_loss)

    if args.annotator_type != 'real':
        val_estimation_error = (val_total_estimation_error).detach().numpy() / float(val_num_error)
        return val_loss, val_estimation_error
    else:
        return val_loss


def compute_transition_probs(model_T, train_loader, val_loader, Num_train, Num_val, args, device):
    
    K = args.num_classes
    R = args.R

    Train_T = torch.zeros((Num_train, R, K, K)).to(device)
    for batch_idx, (indexes, batch_x, batch_y, trusted_label_1, trusted_label_2, annot, noisy_label) in enumerate(train_loader):
        with torch.no_grad():
            model_T.eval()
            batch_x = batch_x.to(device)
            batch_T = model_T(batch_x)
            Train_T[indexes] = batch_T
    
    return Train_T


###################################### Update base model ######################################

# pairwise LRT update

def pairwise_LRT_update(model_1, model_2, model_T, dataloader, args, thr_Omega_trusted, thr_Omega_new, device, logger):

    K = args.num_classes
    R = args.R
    num_batch = len(dataloader)

    model_1.eval()
    model_2.eval()
    
    update_idx_list_1 = []
    update_idx_list_2 = []
    update_labels_list_1 = []
    update_labels_list_2 = []

    Num_new_1 = 0
    Num_new_2 = 0
    Num_new_correct_1 = 0
    Num_new_correct_2 = 0

    Num_del_1 = 0
    Num_del_2 = 0
    Num_del_correct_1 = 0
    Num_del_correct_2 = 0
    
    for batch_idx, (indexes, batch_x, batch_y, trusted_label_1, trusted_label_2, annot, noisy_label) in enumerate(dataloader):

        batch_size = batch_x.shape[0]
        
        if torch.cuda.is_available:
            batch_x = batch_x.to(device)
            annot = annot.to(device) # (n, R)
        
        with torch.no_grad():
            probs_prior_1 = model_1(batch_x) # (n, K)
            probs_prior_2 = model_2(batch_x) # (n, K)

            batch_T = model_T(batch_x) # (n, R, K, K)
            
            annot_one_hot = torch.zeros((batch_size, R, K)).to(device) # (n, R, K)
            annot_one_hot = annot_one_hot.view(batch_size * R, K) # (n * R, K)
            mask = (annot.view(-1) != -1).to(device) # (n * R, )
            annot_one_hot[mask] = F.one_hot(annot.view(-1)[mask], K).float()
            annot_one_hot = annot_one_hot.view(batch_size, R, K) # (n, R, K)

        probs_prior_1 = probs_prior_1.cpu().detach().numpy() # (n, K)
        probs_prior_2 = probs_prior_2.cpu().detach().numpy() # (n, K)
        batch_T = batch_T.cpu().detach().numpy() # (n, R, K, K)
        annot_one_hot = annot_one_hot.cpu().detach().numpy() # (n, R, K)
        
        LRT_criterion_1 = np.zeros((batch_size, K))
        LRT_criterion_2 = np.zeros((batch_size, K))
        for i in range(batch_size):
            for j in range(K):
                LRT_criterion_1[i, j] = probs_prior_1[i, j] * np.prod(np.power(batch_T[i, :, j, :], annot_one_hot[i, :, :]))
                LRT_criterion_2[i, j] = probs_prior_2[i, j] * np.prod(np.power(batch_T[i, :, j, :], annot_one_hot[i, :, :]))  

            # normalize
            sum_temp_1 = np.sum(LRT_criterion_1[i, :])
            if sum_temp_1 > 0:
                LRT_criterion_1[i, :] /= sum_temp_1

            sum_temp_2 = np.sum(LRT_criterion_2[i, :])
            if sum_temp_2 > 0:
                LRT_criterion_2[i, :] /= sum_temp_2
        
        # model 1
        top_2_values_1, top_2_idx_1 = torch.topk(torch.from_numpy(LRT_criterion_1), k=2, dim=1) # (n, 2)
        LRT_ratios_1 = top_2_values_1[:, 0] / (top_2_values_1[:, 1] + 1e-10) # (n, )
        top_2_values_1 = top_2_values_1.numpy()
        top_2_idx_1 = top_2_idx_1.numpy()

        # model 2
        top_2_values_2, top_2_idx_2 = torch.topk(torch.from_numpy(LRT_criterion_2), k=2, dim=1) # (n, 2)
        LRT_ratios_2 = top_2_values_2[:, 0] / (top_2_values_2[:, 1] + 1e-10) # (n, )
        top_2_values_2 = top_2_values_2.numpy()
        top_2_idx_2 = top_2_idx_2.numpy()
        
        trusted_label_1 = trusted_label_1.numpy() # (n, )
        trusted_label_2 = trusted_label_2.numpy() # (n, )
        indexes = indexes.numpy()


        # for j in top_k_idx:
        for j in range(batch_size):

            # model 1: update trusted_label_1

            condition_1 = (LRT_ratios_1[j] > thr_Omega_new)
            condition_2 = (trusted_label_1[j] != -1) and (LRT_criterion_1[j, trusted_label_1[j]] < top_2_values_1[j, 0] * thr_Omega_trusted) 
            
            if condition_1:
                update_idx_list_1.extend([indexes[j]])
                update_labels_list_1.extend([top_2_idx_1[j, 0]])

                Num_new_1 += 1
                Num_new_correct_1 += (top_2_idx_1[j, 0] == batch_y.detach().numpy()[j])
            
            elif condition_2:
                update_idx_list_1.extend([indexes[j]])
                update_labels_list_1.extend([-1])

                Num_del_1 += 1
                Num_del_correct_1 += (trusted_label_1[j] != batch_y[j])
            
            # model 2: update trusted_label_2

            condition_1 = (LRT_ratios_2[j] > thr_Omega_new)
            condition_2 = (trusted_label_2[j] != -1) and (LRT_criterion_2[j, trusted_label_2[j]] < top_2_values_2[j, 0] * thr_Omega_trusted) 
            
            if condition_1:
                update_idx_list_2.extend([indexes[j]])
                update_labels_list_2.extend([top_2_idx_2[j, 0]])

                Num_new_2 += 1
                Num_new_correct_2 += (top_2_idx_2[j, 0] == batch_y.detach().numpy()[j])
            
            elif condition_2:
                update_idx_list_2.extend([indexes[j]])
                update_labels_list_2.extend([-1])

                Num_del_2 += 1
                Num_del_correct_2 += (trusted_label_2[j] != batch_y[j])


    update_idx_1 = np.array(update_idx_list_1).astype(int)
    update_idx_2 = np.array(update_idx_list_2).astype(int)

    update_labels_1 = np.array(update_labels_list_1)
    update_labels_2 = np.array(update_labels_list_2)

    if Num_new_1 > 0:
        logger.info("Model 1: ACC of NEW trusted labels: {} (Num: {})".format(Num_new_correct_1 / Num_new_1, Num_new_1))
    if Num_del_1 > 0:
        logger.info("Model 1: ACC of DELETED trusted labels: {} (Num: {})".format(Num_del_correct_1 / Num_del_1, Num_del_1))
    
    if Num_new_2 > 0:
        logger.info("Model 2: ACC of NEW trusted labels: {} (Num: {})".format(Num_new_correct_2 / Num_new_2, Num_new_2))
    if Num_del_2 > 0:
        logger.info("Model 2: ACC of DELETED trusted labels: {} (Num: {})".format(Num_del_correct_2 / Num_del_2, Num_del_2))

    return update_idx_1, update_idx_2, update_labels_1, update_labels_2

# Training base model

def train_base(model_1, model_2, train_loader, epoch, optimizer_1, optimizer_2, scheduler_1, scheduler_2, alg_options, args):

    device = alg_options['device']
    K = args.num_classes
    R = args.R

    train_total_loss_1 = 0
    train_total_loss_2 = 0

    train_total_correct_1 = 0 
    train_total_correct_2 = 0 

    train_num = 0
    total_batch = len(train_loader)

    loss_fn = nn.NLLLoss()

    for batch_idx, (indexes, batch_x, batch_y, trusted_label_1,  trusted_label_2, annot, noisy_label) in enumerate(train_loader):

        batch_size = batch_x.shape[0]
        train_num += batch_size
        
        if torch.cuda.is_available():
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            trusted_label_1 = trusted_label_1.to(device)
            trusted_label_2 = trusted_label_2.to(device)
            annot = annot.to(device)
        
        # -------- model 1 --------
        model_1.train()
        probs_1 = model_1(batch_x) # (batch_size, K)

        # ACC
        y_hat_1 = torch.max(probs_1, 1)[1]

        batch_correct_1 = (y_hat_1 == batch_y).sum()
        train_total_correct_1 += batch_correct_1

        # loss: use trusted_label_2 (!= -1)
        mask = (trusted_label_2 != -1)
        probs_1_mask = probs_1[mask]
        loss_1 = loss_fn(torch.log(probs_1_mask + 1e-10), trusted_label_2[mask])

        batch_loss_1 = loss_1.item()
        train_total_loss_1 += batch_loss_1 *  batch_size # total loss on the batch

        optimizer_1.zero_grad()
        loss_1.backward()
        optimizer_1.step()
        optimizer_1.zero_grad()

        # -------- model 2 --------
        model_2.train()
        probs_2 = model_2(batch_x) # (batch_size, K)

        # ACC
        y_hat_2 = torch.max(probs_2, 1)[1]

        batch_correct_2 = (y_hat_2 == batch_y).sum()
        train_total_correct_2 += batch_correct_2

        # loss: use trusted_label_1 (!= -1)
        mask = (trusted_label_1 != -1)
        probs_2_mask = probs_2[mask]
        loss_2 = loss_fn(torch.log(probs_2_mask + 1e-10), trusted_label_1[mask])

        batch_loss_2 = loss_2.item()
        train_total_loss_2 += batch_loss_2 *  batch_size # total loss on the batch

        optimizer_2.zero_grad()
        loss_2.backward()
        optimizer_2.step()
        optimizer_2.zero_grad()

        # -------- Print --------
        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch [%d], Iter [%d/%d], Training ACC_1: %.4F, Training ACC_2: %.4F, Loss_1: %.4f, Loss_2: %.4f'
                % (epoch + 1 + args.n_epoch_burn, batch_idx + 1, total_batch, batch_correct_1/batch_size, batch_correct_2/batch_size, batch_loss_1, batch_loss_2))
        
    scheduler_1.step()
    scheduler_2.step()

    train_acc_1 = float(train_total_correct_1) / float(train_num)
    train_acc_2 = float(train_total_correct_2) / float(train_num)

    train_loss_1 = float(train_total_loss_1) / float(train_num)
    train_loss_2 = float(train_total_loss_2) / float(train_num)

    return train_acc_1, train_acc_2, train_loss_1, train_loss_2


# Validation

def eval_base(val_loader, model_1, model_2, alg_options, args):
    device = alg_options['device']

    K = args.num_classes
    R = args.R

    val_num = 0

    val_total_loss_1 = 0
    val_total_loss_2 = 0

    val_total_correct_1 = 0 
    val_total_correct_2 = 0 

    loss_fn = nn.NLLLoss()
    
    with torch.no_grad():

        for _, (indexes, batch_x, batch_y, trusted_label_1, trusted_label_2, annot, noisy_label) in enumerate(val_loader):

            val_num += batch_x.shape[0]
            batch_size = batch_x.shape[0]

            if torch.cuda.is_available:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device) # (batch_size,)
                trusted_label_1 = trusted_label_1.to(device)
                trusted_label_2 = trusted_label_2.to(device)
                annot = annot.to(device) # (batch_size, R)
            
            # -------- model 1 --------
            model_1.eval()
            probs_1 = model_1(batch_x) # (batch_size, K)

            # ACC
            y_hat_1 = torch.max(probs_1, 1)[1]

            batch_correct_1 = (y_hat_1 == batch_y).sum()
            val_total_correct_1 += batch_correct_1

            # loss: use trusted_label_2 (!= -1)
            mask = (trusted_label_2 != -1)
            probs_1_mask = probs_1[mask]
            loss_1 = loss_fn(torch.log(probs_1_mask + 1e-10), trusted_label_2[mask])

            batch_loss_1 = loss_1.item()
            val_total_loss_1 += batch_loss_1 *  batch_size # total loss on the batch

            # -------- model 2 --------
            model_2.eval()
            probs_2 = model_2(batch_x) # (batch_size, K)

            # ACC
            y_hat_2 = torch.max(probs_2, 1)[1]

            batch_correct_2 = (y_hat_2 == batch_y).sum()
            val_total_correct_2 += batch_correct_2

            # loss: use trusted_label_1 (!= -1)
            mask = (trusted_label_1 != -1)
            probs_2_mask = probs_2[mask]
            loss_2 = loss_fn(torch.log(probs_2_mask + 1e-10), trusted_label_1[mask])

            batch_loss_2 = loss_2.item()
            val_total_loss_2 += batch_loss_2 *  batch_size # total loss on the batch

        val_acc_1 = float(val_total_correct_1) / float(val_num)
        val_acc_2 = float(val_total_correct_2) / float(val_num)

        val_loss_1 = float(val_total_loss_1) / float(val_num)
        val_loss_2 = float(val_total_loss_2) / float(val_num)
    
    return val_acc_1, val_acc_2, val_loss_1, val_loss_2




  
