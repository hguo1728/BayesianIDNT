import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch.hub as hub
import torchvision.models as models
from torch.autograd import Variable
from torchvision import transforms
from torch.distributions import RelaxedOneHotCategorical, RelaxedBernoulli


class FCNN(nn.Module):

    def __init__(self, input_dim=8192, K=10):
        super(FCNN, self).__init__()

        self.input_dim = input_dim
        self.K = K

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, K)
        self.dropout = nn.Dropout(0.5)
        self.T_revision = nn.Linear(self.K, self.K, False)
    
    def forward(self, x, revision=False):
        correction = self.T_revision.weight

        x = x.reshape(x.size()[0], self.input_dim)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        x = F.softmax(x, dim=1)

        if revision == True:
            return  x, correction
        else:
            return  x
        
    def copy_structure(self):
        return FCNN()


class FCNN_F(nn.Module):

    def __init__(self, input_dim=8192, K=10):
        super(FCNN_F, self).__init__()

        self.input_dim = input_dim
        self.K = K

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2_bayes = nn.Linear(128, K*K)
        self.dropout = nn.Dropout(0.5)
        
    
    def forward(self, x):

        x = x.reshape(x.size()[0], self.input_dim)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2_bayes(x)

        x = x.reshape(x.size(0), self.K, self.K)

        x = F.softmax(x, dim=2)

        return  x
        
    def copy_structure(self):
        return FCNN()





class FCNN_T(nn.Module):

    def __init__(self, R=59, K=8):
        super(FCNN_T, self).__init__()
        self.R = R
        self.K = K

        self.fc1_1 = nn.Linear(512, 128)
        self.fc1_2 = nn.Linear(512, 128)
        
        self.out_1  = nn.Linear(128, self.K * self.K)
        self.out_2  = nn.Linear(128, self.K * self.R)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):

        x = self.avgpool(x) # (n, 512, 4, 4) -> (n, 512, 1, 1)
        x = x.squeeze(2).squeeze(2) 

        out1 = F.relu(self.fc1_1(x)) # 512 -> 128 
        out1 = self.out_1(out1) # 128 -> K * K
        out1 = out1.reshape(out1.size(0), self.K, self.K) # (n, K, K)

        out2 = F.relu(self.fc1_2(x)) # 512 -> 128 
        out2 = self.out_2(out2) # 128 -> R * K
        out2 = out2.reshape(out2.size(0), self.R, self.K) # (n, R, K)

        out = out1[:, None, :, :] + out2[:, :, None, :]
        prob = F.softmax(out, dim=3) # (n, R, K, K)

        return prob
    
    def get_pretrained_weights(self, model_base):
        model_base.eval()
        for name, module in self.named_children():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)







class FCNN_BatchNorm(nn.Module):

    def __init__(self, input_dim=124, K=10):
        super(FCNN_BatchNorm, self).__init__()
        self.input_dim = input_dim
        self.K = K

        self.bn1 = nn.BatchNorm1d(input_dim,affine=False)
        self.fc1 = nn.Linear(input_dim, 128)

        self.bn2 = nn.BatchNorm1d(128,affine=False)
        self.fc2 = nn.Linear(128, K)

        self.dropout = nn.Dropout(0.5)

        self.T_revision = nn.Linear(K, K, False)
    
    def forward(self, x, revision=False):
        correction = self.T_revision.weight

        x = x.reshape(x.size()[0], self.input_dim)

        x = self.bn1(x)
        x = self.dropout(F.relu(self.fc1(x)))

        x =self.bn2(x)
        x = self.fc2(x)

        x = F.softmax(x, dim=1)

        if revision == True:
            return  x, correction
        else:
            return  x
        
    
    def copy_structure(self):
        return FCNN_BatchNorm()

class FCNN_BatchNorm_F(nn.Module):

    def __init__(self, input_dim=124, K=10):
        super(FCNN_BatchNorm_F, self).__init__()
        self.input_dim = input_dim
        self.K = K

        self.bn1 = nn.BatchNorm1d(input_dim,affine=False)
        self.fc1 = nn.Linear(input_dim, 128)

        self.bn2 = nn.BatchNorm1d(128,affine=False)
        self.fc2_bayes = nn.Linear(128, K*K)

        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = x.reshape(x.size()[0], self.input_dim)

        x = self.bn1(x)
        x = self.dropout(F.relu(self.fc1(x)))

        x =self.bn2(x)
        x = self.fc2_bayes(x)

        x = x.reshape(x.size(0),self.K,self.K)

        x = F.softmax(x, dim=2)
        return x
    
    def copy_structure(self):
        return FCNN_BatchNorm()







class FCNN_BatchNorm_T(nn.Module):

    def __init__(self, input_dim=124, K=10, R=44):
        super(FCNN_BatchNorm_T, self).__init__()
        self.input_dim = input_dim
        self.R = R
        self.K = K

        self.bn1 = nn.BatchNorm1d(input_dim,affine=False)
        self.bn2 = nn.BatchNorm1d(128,affine=False)

        self.fc1_1 = nn.Linear(input_dim, 128)
        self.fc1_2 = nn.Linear(input_dim, 128)

        self.out_1  = nn.Linear(128, self.K * self.K)
        self.out_2  = nn.Linear(128, self.K * self.R)

        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        
        out1 = self.bn1(x)
        out1 = self.dropout(F.relu(self.fc1_1(out1))) # 124 -> 128
        out1 = self.bn2(out1)
        out1 = self.out_1(out1) # 128 -> K * K
        out1 = out1.reshape(out1.size(0), self.K, self.K) # (n, K, K)

        out2 = self.bn1(x)
        out2 = self.dropout(F.relu(self.fc1_2(out2))) # 124 -> 128
        out2 = self.bn2(out2)
        out2 = self.out_2(out2) # 128 -> R * K
        out2 = out2.reshape(out2.size(0), self.R, self.K) # (n, R, K)

        out = out1[:, None, :, :] + out2[:, :, None, :]
        prob = F.softmax(out, dim=3) # (n, R, K, K)

        return prob
    
    def get_pretrained_weights(self, model_base):
        model_base.eval()
        for name, module in self.named_children():
            if name == 'fc1_1' or name == 'fc1_2':
                dict = model_base.fc1.state_dict()
                module.load_state_dict(dict)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
