# import os

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import argparse

import numpy as np
from numpy.random import default_rng
import random
from sklearn.model_selection import train_test_split
from datetime import datetime

from utils.synthetic import *
from Data_load.mnist import *
from Data_load.cifar10 import *
from Data_load.cifar100 import *
from Data_load.transformer import *
from Data_load.labelme import *

from train.train_ours import *




parser = argparse.ArgumentParser()

########################################

parser.add_argument('--model_running', type = str, help='training methods', default='CE') 
parser.add_argument('--dataset',type=str,help='synthetic, mnist, fmnist, cifar10, cifar100, cifar10n, labelme',default='cifar10')

parser.add_argument('--Reg_lambda',type=float,help='lambda: regularization parameter (Trace_Reg, GeoCrowdNet, Ours)',default=0)
parser.add_argument('--GeoCrowdNet_method',type=str,help='GeoCrowdNet method: F or W',default="W")

######### get noisy labels
parser.add_argument('--error_rate_type',type=str,help='error rates types: low, mid, high',default="low") 
parser.add_argument('--error_rate',type=float,help='error rate',default=0.2) 
parser.add_argument('--annotator_type',type=str,help='instance-dependent, real',default='instance-dependent')
parser.add_argument('--R',type=int,help='No of annotators',default=30) 
parser.add_argument('--l',type=int,help='number of annotations per sample or number of samples per annotators',default=1) 
parser.add_argument('--p',type=float,help='prob. that an annotator label a sample',default=0.1) 

######### file management
parser.add_argument('--log_folder',type=str,help='log folder path',default='results/mnist/ours/low')
parser.add_argument('--file_name',type=str,help='file name',default='_mnist_5_low.txt')

######### Basic setting 
parser.add_argument('--seed',type=int,help='Random seed',default=1) 
parser.add_argument('--device',type=int,help='GPU device number',default=0)
parser.add_argument('--n_trials',type=int,help='No of trials',default=5)
parser.add_argument('--print_freq', type=int, default=50)




######### Dataset and annotations
# ------------ data ------------ 
parser.add_argument('--split_percentage', type = float, help = 'train and validation', default=0.9) 
parser.add_argument('--norm_std', type = float, help = 'distribution ', default=0.1)
parser.add_argument('--num_classes', type = int, help = 'num_classes', default=10) 
parser.add_argument('--feature_size', type = int, help = 'the size of feature_size', default=784) 


######### Learning parameters -- base
parser.add_argument('--optimizer', type = str, default='SGD') # not used
parser.add_argument('--num_workers', type = int, default=3, help='how many subprocesses to use for data loading')
parser.add_argument('--learning_rate',type=float,help='Learning rate',default=0.01)
parser.add_argument('--batch_size',type=int,help='Batch Size',default=128)
parser.add_argument('--n_epoch',type=int,help='Number of Epochs: base',default=100)
parser.add_argument('--n_epoch_burn',type=int,help='Number of Epochs: base (burn in)',default=20) 
parser.add_argument('--weight_decay', type=float, help='l2', default=5e-4)
parser.add_argument('--momentum', type=int, help='momentum', default=0.9) 
parser.add_argument('--thr', type = float, help = 'threshold for collecting trusted examples (warm up)', default=0.7)
# LRT:
parser.add_argument('--trusted_ratio_init', type = float, help = '(max) ratio of trusted data points', default=0.6)
parser.add_argument('--epoch_start_LRT',type=int,help='start pairwise LRT test',default=0)
parser.add_argument('--freq_LRT',type=float,help='fequency of LRT',default=1)
parser.add_argument('--trusted_ratio_max', type = float, help = '(max) ratio of trusted data points (LRT)', default=0.9) 

parser.add_argument('--thr_Omega_new', type = float, help = 'LRT: thr for obtaining new trusted labeld', default=10)
parser.add_argument('--thr_increment_LRT', type = float, help = 'LRT: thr increment after each epoch', default=1)

parser.add_argument('--thr_Omega_trusted', type = float, help = 'LRT: thr for deleting trusted labels', default=1)
parser.add_argument('--thr_Omega_trusted_end', type = float, help = 'LRT: thr for deleting trusted labels (end)', default=10)

parser.add_argument('--start_LRT', type = float, help = 'LRT: thr start ratio', default=10)
parser.add_argument('--end_LRT', type = float, help = 'LRT: thr end ratio', default=100)

######### Learning parameters -- Transition
parser.add_argument('--learning_rate_T',type=float,help='Learning rate (transition)',default=0.001)
parser.add_argument('--n_epoch_T_init',type=int,help='Number of Epochs: Transition (init)',default=30)
parser.add_argument('--n_epoch_T_fine_tune',type=int,help='Number of Epochs: Transition (fine tune)',default=1)
parser.add_argument('--freq_update_T',type=float,help='fequency of updating the transition net',default=5)
parser.add_argument('--sigma0', type = float, help = 'prior para: sigma_0^2', default = 0.000002)
parser.add_argument('--sigma1', type = float, help = 'prior para: sigma_1^2', default = 0.04)
parser.add_argument('--lambdan', type = float, help = 'prior para: lambda_n', default = 0.0001)


# Parser
args=parser.parse_args()

# Setting GPU and cuda settings
torch.backends.cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.autograd.set_detect_anomaly(True)

# Log file settings
time_now = datetime.now()
time_now.strftime("%b-%d-%Y")
log_file_name = args.log_folder+'/log_' + str(time_now.strftime("%b-%d-%Y")) + '.txt'
result_file = args.log_folder+'/result_' + str(time_now.strftime("%b-%d-%Y")) + '.txt'


        
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(message)s')

fh = logging.FileHandler(log_file_name)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

ch = logging.StreamHandler()
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)




########################################################################################
#                                                                                      #
#                                  Load Dataset                                        #
#                                                                                      #
########################################################################################

def load_data(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    EM = False
    if args.model_running == 'EM':
        EM = True

    if args.dataset=='mnist':
        train_data = mnist_dataset(
                                train=True, 
                                transform=transform_train(args.dataset), target_transform=transform_target, 
                                split_percentage=0.9, random_seed=args.seed,
                                args=args, logger=logger, num_class=args.num_classes, 
                                EM=EM
                                     )
        
        val_data = mnist_dataset(
                                train=False, 
                                transform=transform_test(args.dataset), target_transform=transform_target, 
                                split_percentage=0.9, random_seed=args.seed,
                                args=args, logger=logger, num_class=args.num_classes, 
                                EM=EM
                                     )
        
        test_data = mnist_test_dataset(
                                transform=transform_test(args.dataset), 
                                target_transform=transform_target
                                     )

    if args.dataset=='cifar10':
        train_data = cifar10_dataset(
                                train=True, 
                                transform=transform_train(args.dataset), target_transform=transform_target, 
                                split_percentage=0.9, random_seed=args.seed,
                                args=args, logger=logger, num_class=args.num_classes, 
                                EM=EM
                                     )
        
        val_data = cifar10_dataset(
                                train=False, 
                                transform=transform_test(args.dataset), target_transform=transform_target, 
                                split_percentage=0.9, random_seed=args.seed,
                                args=args, logger=logger, num_class=args.num_classes, 
                                EM=EM
                                     )
        
        test_data = cifar10_test_dataset(
                                transform=transform_test(args.dataset), 
                                target_transform=transform_target
                                     )
    
    if args.dataset=='cifar100':
        train_data = cifar100_dataset(
                                train=True, 
                                transform=transform_train(args.dataset), target_transform=transform_target, 
                                split_percentage=0.9, random_seed=args.seed,
                                args=args, logger=logger, num_class=args.num_classes, 
                                EM=EM
                                     )
        
        val_data = cifar100_dataset(
                                train=False, 
                                transform=transform_test(args.dataset), target_transform=transform_target, 
                                split_percentage=0.9, random_seed=args.seed,
                                args=args, logger=logger,num_class=args.num_classes, 
                                EM=EM
                                     )
        
        test_data = cifar100_test_dataset(
                                transform=transform_test(args.dataset), 
                                target_transform=transform_target
                                     )

    if args.dataset=='labelme':
        train_data = labelme_dataset(
                                train=True, 
                                transform=transform_train(args.dataset), target_transform=transform_target, 
                                split_percentage=0.9, random_seed=args.seed,
                                args=args, logger=logger, num_class=args.num_classes, 
                                EM=EM
                                     )
        
        test_data = labelme_test_dataset(
                                transform=transform_test(args.dataset), 
                                target_transform=transform_target, test=True
                                     )
        
        if args.model_running != 'GCE': 
            val_data = labelme_test_dataset(
                                    transform=transform_test(args.dataset), 
                                    target_transform=transform_target, test=False
                                     )
        else:
            val_data = labelme_test_dataset(
                                    transform=transform_test(args.dataset), 
                                    target_transform=transform_target, test=False, return_idx=True
                                     )
    
    return train_data, val_data, test_data




########################################################################################
#                                                                                      #
#                                        Main                                          #
#                                                                                      #
########################################################################################


def main():

    ######################################## setup ########################################

    rng = default_rng()

    # Data logging variables
    
    fileid = open(result_file, "w")
    fileid.write('#########################################################\n')
    fileid.write(str(time_now))
    fileid.write('\n')
    fileid.write('Trial#\t')
    fileid.write(args.model_running)
    fileid.write('\n')

    annotators_sel = range(args.R)
    alg_options = { 
        'device':device,
        'loss_function_type':'cross_entropy'}


    if args.dataset == "mnist":

        args.num_classes = 10
        args.feature_size = 28 * 28

        args.learning_rate = 0.01
        args.batch_size = 128
        args.n_epoch_burn = 20   
        args.n_epoch = 80 

        args.learning_rate_T = 1e-3
        args.n_epoch_T_init = 10 
        args.n_epoch_T_fine_tune = 10 

        args.lambdan = 5e-9
        args.sigma0 = 2e-7
        args.sigma1 = 5e-1

    if args.dataset == "cifar10":

        args.num_classes = 10
        args.feature_size = 3 * 32 * 32

        args.learning_rate = 0.001
        args.batch_size = 128
        args.n_epoch_burn = 30  
        args.n_epoch = 120 

        args.n_epoch_T_init = 20 
        args.n_epoch_T_fine_tune = 10 

        args.lambdan = 5e-9
        args.sigma0 = 2e-7
        args.sigma1 = 5e-1

        args.Reg_lambda = 0
        
        
    if args.dataset == "cifar100":

        
        args.num_classes = 100
        args.feature_size = 3 * 32 * 32

        args.learning_rate = 0.001
        args.batch_size = 128
        args.n_epoch_burn = 30   
        args.n_epoch = 150 

        args.learning_rate_T = 1e-3
        args.n_epoch_T_init = 20 
        args.n_epoch_T_fine_tune = 10 

        args.lambdan = 5e-9
        args.sigma0 = 2e-7
        args.sigma1 = 5e-1

        args.Reg_lambda = 0
    

    if args.dataset == "labelme":

        args.num_classes = 8
        args.feature_size = 8192
        args.R = 59

        args.print_freq = 30

        args.batch_size = 128
        args.n_epoch_burn = 20 

        args.n_epoch = 100 
        args.n_epoch_T_init = 20 
        args.n_epoch_T_fine_tune = 10 

        args.lambdan = 5e-9
        args.sigma0 = 2e-7
        args.sigma1 = 5e-1


     

    ######################################## n_trials: repeated experiments ########################################
	
    t = args.seed

    logger.info(" ")
    logger.info('--*--*--*--*--*--*--*--*--*--*--*--*--*-- Starting trial ' + str(t) + '_' + datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + '--*--*--*--*--*--*--*--*--*--*--*--*--*--')
    logger.info(" ")

    # ------------------------------ Setup & Load Data ------------------------------

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    # Get the train, validation and test dataset 
    train_data, val_data, test_data = load_data(args)
    
    alg_options['train_dataset']=train_data
    alg_options['val_dataset']=val_data
    alg_options['annotators_sel']=annotators_sel

    # Training sample size
    args.N = train_data.train_data.shape[0]
    
    # Prepare data for training/validation and testing
    train_loader = DataLoader(dataset=train_data,
                            batch_size=args.batch_size,
                            num_workers=3,
                            shuffle=True,
                            drop_last=False, 
                            pin_memory=False) 
    
    train_loader_no_shuffle = DataLoader(dataset=train_data,
                            batch_size=args.batch_size,
                            num_workers=3,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=False)
    
    
    test_loader = DataLoader(dataset=test_data,
                            batch_size=args.batch_size,
                            num_workers=3,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=False)
    
    val_loader = DataLoader(dataset=val_data,
                            batch_size=args.batch_size,
                            num_workers=3,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=False)
    
    val_loader_shuffle = DataLoader(dataset=val_data,
                                batch_size=args.batch_size,
                                num_workers=3,
                                shuffle=True,
                                drop_last=False,
                                pin_memory=False)
    
                            
    alg_options['train_loader_no_shuffle'] = train_loader_no_shuffle				  
    alg_options['train_loader'] = train_loader
    alg_options['val_loader'] = val_loader
    alg_options['val_loader_shuffle'] = val_loader_shuffle 
    alg_options['test_loader']= test_loader
                        
    # ------------------------------ Run Algorithm ------------------------------

    fileid.write(str(args.seed)+'\t')


    t1 = time.time()

    if args.model_running == "ours":

        test_acc_1, test_acc_2 = train_ours(args, alg_options, logger, t)


        print("############### ACC_1 (Ours): {} ###############".format(test_acc_1))
        print("############### ACC_2 (Ours): {} ###############".format(test_acc_2))
        fileid.write("%.4f\t" %(test_acc_1))
        fileid.write("%.4f\t" %(test_acc_2))
        fileid.write('\n')

    t2 = time.time()
    print("Running time:", t2-t1)
    
    logger.removeHandler(fh)
    logger.removeHandler(ch)
			

if __name__ == '__main__':
	main()
	

