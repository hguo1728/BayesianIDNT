import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch.hub as hub
import torchvision.models as models
from torch.autograd import Variable
from torchvision import transforms
from torch.distributions import RelaxedOneHotCategorical, RelaxedBernoulli

from models.LeNet import *
from models.ResNet import *
from models.FCNN import *


####################################################################################################
# CrowdLayer (Rodrigues_Pereira, AAAI 2018)
# TraceReg (Tanno_Saeedi_others, CVPR 2019)
# GeoCrowdNet(F) and GeoCrowdNet(M) (Ibrahim_Nguyen_others, 2023 ICLR)

# 'MV': f(sigma_r) = W^r * sigma, where sigma denotes the output of the base model
# Initialization: identity


class CrowdNet(nn.Module):
    def __init__(self, R, K, base_model_type, init_method='identity', A_init=None):
        
        """
        * R: num of annotators
        * K: num of classes
        * base_model_type: base model (according to different dataset)
        * init_method: initialization of crowd layer
        * A_init: if init_method == 'mle_based'
        """
        
        super(CrowdNet,self).__init__()
        
        # base model
        if base_model_type =='lenet':
            self.base_model = Lenet()
            
        elif base_model_type =='resnet18':
            self.base_model = ResNet18(K)
            
        elif base_model_type =='resnet34':
            self.base_model = ResNet34(K)

        elif base_model_type == 'FCNN':
            self.base_model = FCNN()

            
        # crowd layer
        if init_method=='identity':
            self.crowd_layer = nn.Parameter(torch.stack([torch.eye(K) for _ in range(R)]), requires_grad=True) # (R, K, K)
            
        elif init_method=='mle_based':
            self.crowd_layer = nn.Parameter(torch.stack([A_init[r] for r in range(R)]), requires_grad=True)
            
        else:
            self.crowd_layer = nn.Parameter(torch.stack([torch.eye(K) for _ in range(R)]), requires_grad=True)
        

    def forward(self,x):
        x = self.base_model(x)
        A = F.softmax(self.crowd_layer, dim=1) # (R, K, K) -- (num_annotator, noisy_label, true_label)
        y = torch.einsum('ij, bkj -> ibk', x, A) # (n, K) * (R, K, K) -> (n, R, K)
        return (x, y, A) # (base_model_preds, combined_model_preds, transition matrices)



####################################################################################################
# Rodrigues & Pereira (2017); Doctor_Net

class Weights(nn.Module):

    def __init__(self, R):
        super(Weights, self).__init__()
        self.weights = nn.Parameter(torch.ones(R)/R, requires_grad=True)
        
    def forward(self, x=None):
        return self.weights/self.weights.sum(0)

class doctor_net(nn.Module):
    
    def __init__(self, R, K, base_model_type):
        super(doctor_net, self).__init__()

        self.R = R
        self.K = K
        
        # backbone
        if base_model_type =='lenet':
            self.backbone = Lenet()
            self.features = nn.Sequential(
                self.backbone.conv1,
                nn.ReLU(),
                nn.MaxPool2d(2),

                self.backbone.conv2,
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Flatten(1),
                nn.Linear(400, 120),
                nn.Linear(120, 84),
            )
            
        elif base_model_type =='resnet18':
            self.backbone = ResNet18(K)
            self.features = nn.Sequential(
                self.backbone.conv1,
                self.backbone.bn1,
                nn.ReLU(),
                self.backbone.layer1,
                self.backbone.layer2,
                self.backbone.layer3,
                self.backbone.layer4,
                self.backbone.avgpool,
                nn.Flatten(1),
            )
            
        elif base_model_type =='resnet34':
            self.backbone = ResNet34(K)
            self.features = nn.Sequential(
                self.backbone.conv1,
                self.backbone.bn1,
                nn.ReLU(),
                self.backbone.layer1,
                self.backbone.layer2,
                self.backbone.layer3,
                self.backbone.layer4,
                self.backbone.avgpool,
                nn.Flatten(1),
            )
            
        elif base_model_type == 'FCNN':
            self.features = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(8192, 128),
                nn.ReLU(),
                nn.Dropout(0.5)
            )

        if base_model_type =='lenet':
            self.fc = nn.Linear(84, K * R)
        elif base_model_type =='FCNN':
            self.fc = nn.Linear(128, K * R)
        else:
            self.fc = nn.Linear(512, K * R)

        self.weights = Weights(R)


    def forward(self, x, pred=False, weight=False):
        x = self.features(x)
        x = self.fc(x)
        x = x.reshape(x.size(0), self.R, self.K)
        x = x.softmax(2) # (n, R, K)
        weights = self.weights()

        if weight:
            x = x * weights[None, :, None]
        
        if pred:
            x = torch.sum(x, 1)
            return x
        else:
            return x, weights



####################################################################################################
# Max-MIG (Cao_Xu_others, ICLR 2019)

class Right_Net(nn.Module):
    """
    right neural net for max-mig
    """

    def __init__(self, R, K, prior, confusion_init, device):
        super(Right_Net, self).__init__()
        self.priority = prior.to(device)

        for i in range(R):
            m_name = "fc" + str(i+1)
            self.add_module(m_name,nn.Linear(K, K, bias=False))
        
        confusion_init = torch.from_numpy(confusion_init).float().to(device)
        self.weights_init(confusion_init)

    def forward(self, x, left_p, prior = 0, type=0):
        """
        input
        * x: annot_one_hot, (n, R, K)
        """
        out = 0
        for name, module in self.named_children():
            if name == 'p':
                continue
            index = int(name[2:])
            out += module(x[:, index-1, :]) # (n, K) -> (K, K) linear layer -> (n, K) 

        if type == 1 :
            out += torch.log(left_p+0.001) + torch.log(self.priority)

        elif type == 2 : # this is the type the anthors used
            out += torch.log(self.priority)

        elif type == 3 :
            out += torch.log(left_p + 0.001)

        return torch.nn.functional.softmax(out,dim=1)

    def weights_init(self, confusion_init):
        for name, module in self.named_children():
            if name == 'p':
                module.weight.data = self.priority
                continue
            index = int(name[2:])
            module.weight.data = torch.log(confusion_init[index - 1] + 0.0001)



####################################################################################################
# CoNAL (Chu_Ma_Wang, AAAI 2021)

class CoNAL(nn.Module):
    def __identity_init(self, shape, device):
        out = np.ones(shape) * 0
        if len(shape) == 3:
            for r in range(shape[0]):
                for i in range(shape[1]):
                    out[r, i, i] = 2
        elif len(shape) == 2:
            for i in range(shape[1]):
                out[i, i] = 2
        return torch.Tensor(out).to(device)

    def __init__(self, num_annotators, input_dims, num_class, device, rate=0.5, conn_type='MW', backbone_model=None, user_feature=None
                 , common_module='simple', num_side_features=None, nb=None, u_features=None,
                 v_features=None, u_features_side=None, v_features_side=None, input_dim=None, emb_dim=None, hidden=None, gumbel_common=False):
        super(CoNAL, self).__init__()
        self.num_annotators = num_annotators
        self.conn_type = conn_type
        self.gumbel_sigmoid = GumbelSigmoid(temp=0.01)

        self.linear1 = nn.Linear(input_dims, 128)

        self.ln1 = nn.Linear(128, 256)
        self.ln2 = nn.Linear(256, 128)

        self.linear2 = nn.Linear(128, num_class)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.rate = rate
        self.kernel = nn.Parameter(self.__identity_init((num_annotators, num_class, num_class), device),
                                   requires_grad=True)

        self.common_kernel = nn.Parameter(self.__identity_init((num_class, num_class), device) ,
                                          requires_grad=True)

        self.backbone_model = None
        if backbone_model =='lenet':
            self.backbone_model = Lenet().to(device)
            
        elif backbone_model =='resnet18':
            self.backbone_model = ResNet18(num_class).to(device)
            
        elif backbone_model =='resnet34':
            self.backbone_model = ResNet34(num_class).to(device)
            
        elif backbone_model == 'FCNN':
            self.backbone_model = FCNN().to(device)
            
        self.common_module = common_module

        if self.common_module == 'simple':
            com_emb_size = 20
            self.user_feature_vec = torch.from_numpy(user_feature).float().to(device)
            self.diff_linear_1 = nn.Linear(input_dims, 128)
            self.diff_linear_2 = nn.Linear(128, com_emb_size)
            self.user_feature_1 = nn.Linear(self.user_feature_vec.size(1), com_emb_size)
            self.bn_instance = torch.nn.BatchNorm1d(com_emb_size, affine=False)
            self.bn_user = torch.nn.BatchNorm1d(com_emb_size, affine=False)
            self.single_weight = nn.Linear(20, 1, bias=False)

    def simple_common_module(self, input):
        instance_difficulty = self.diff_linear_1(input)
        instance_difficulty = self.diff_linear_2(instance_difficulty)

        instance_difficulty = F.normalize(instance_difficulty)
        user_feature = self.user_feature_1(self.user_feature_vec)
        user_feature = F.normalize(user_feature)
        common_rate = torch.einsum('ij,kj->ik', (instance_difficulty, user_feature))
        common_rate = torch.sigmoid(common_rate)
        return common_rate

    def forward(self, input, y=None, mode='train', support=None, support_t=None, idx=None):
        crowd_out = None
        cls_out = self.backbone_model(input)
        
        if mode == 'train':
            x = input.view(input.size(0), -1)
            if self.common_module == 'simple':
                common_rate = self.simple_common_module(x)
            common_prob = torch.einsum('ij,jk->ik', (cls_out, self.common_kernel)) # (n, K) * (K, K) -> (n, K)
            indivi_prob = torch.einsum('ik,jkl->ijl', (cls_out, self.kernel)) # (n, K) * (R, K, K) -> (n, R, K)

            crowd_out = common_rate[:, :, None] * common_prob[:, None, :] + (1 - common_rate[:, :, None]) * indivi_prob   # (n, R, K)
            crowd_out = crowd_out.softmax(2)

        if self.common_module == 'simple' or mode == 'test':
            return cls_out, crowd_out




def gumbel_sigmoid(input, temp):
    return RelaxedBernoulli(temp, probs=input).rsample()


class GumbelSigmoid(nn.Module):
    def __init__(self,
                 temp: float = 0.1,
                 threshold: float = 0.5):
        super(GumbelSigmoid, self).__init__()
        self.temp = temp
        self.threshold = threshold

    def forward(self, input):
        if self.training:
            return gumbel_sigmoid(input, self.temp)
        else:
            return (input.sigmoid() >= self.threshold).float()


