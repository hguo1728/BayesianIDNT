import torch
import numpy as np
import torchvision.transforms as transforms

def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def transform_train(dataset_name):
    
    if dataset_name == 'mnist':
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ),(0.3081, )),
       ])
    
    elif dataset_name == 'cifar10' or dataset_name == 'cifar100':
        transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
        ])
    
    elif dataset_name == 'labelme' or dataset_name == 'music':
        transform = transforms.ToTensor()
    
    return transform


def transform_test(dataset_name):
    
    if dataset_name == 'mnist':
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ),(0.3081, )),
       ])
    
    elif dataset_name == 'cifar10' or dataset_name == 'cifar100':
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
        ])
    
    elif dataset_name == 'labelme'or dataset_name == 'music':
        transform = transforms.ToTensor()
    
    return transform


def transform_target(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target    
