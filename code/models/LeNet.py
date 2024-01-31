import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch.hub as hub
import torchvision.models as models
from torch.autograd import Variable
from torchvision import transforms
from torch.nn.utils import prune


class Lenet(nn.Module):

    def __init__(self, K=10):
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5,stride=1,padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(400, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

        self.K = K
        self.T_revision = nn.Linear(self.K, self.K, False)


    def forward(self, x, revision=False):

        correction = self.T_revision.weight

        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        prob = F.softmax(out, 1)

        if revision == True:
            return prob, correction
        else:
            return prob

    
    def get_feature(self, x):

        output_list = []

        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        output_list.append(out)

        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        output_list.append(out)

        feature_num = len(output_list)
        for i in range(feature_num):
            output_list[i] = output_list[i].reshape(output_list[i].shape[0], -1)

        return output_list
    
    def copy_structure(self):
        return Lenet()







class Lenet_T(nn.Module):

    def __init__(self, R, K):
        super(Lenet_T, self).__init__()
        self.R = R
        self.K = K

        self.conv1 = nn.Conv2d(1, 6, 5,stride=1,padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1_1 = nn.Linear(400, 120)
        self.fc2_1 = nn.Linear(120, 84)

        self.fc1_2 = nn.Linear(400, 120)
        self.fc2_2 = nn.Linear(120, 84)

        self.out_1  = nn.Linear(84, self.K * self.K)
        self.out_2  = nn.Linear(84, self.K * self.R)


    def forward(self, x):

        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1) # 400

        out1 = F.relu(self.fc1_1(out)) # 400 -> 120
        out1 = F.relu(self.fc2_1(out1)) # 120 -> 84
        out1 = self.out_1(out1) # 84 -> K * K
        out1 = out1.reshape(out1.size(0), self.K, self.K) # (n, K, K)

        out2 = F.relu(self.fc1_2(out)) 
        out2 = F.relu(self.fc2_2(out2))
        out2 = self.out_2(out2) # 84 -> R * K
        out2 = out2.reshape(out2.size(0), self.R, self.K) # (n, R, K)

        out = out1[:, None, :, :] + out2[:, :, None, :]
        prob = F.softmax(out, dim=3) # (n, R, K, K)

        return prob
    
    def get_pretrained_weights(self, model_base):
        model_base.eval()
        for name, module in self.named_children():
            if name[:5] == 'conv1':
                dict = model_base.conv1.state_dict()
                module.load_state_dict(dict)
            elif name[:5] == 'conv2':
                dict = model_base.conv2.state_dict()
                module.load_state_dict(dict)
            elif name[:3] == 'fc1':
                dict = model_base.fc1.state_dict()
                module.load_state_dict(dict)
            elif name[:3] == 'fc2':
                dict = model_base.fc2.state_dict()
                module.load_state_dict(dict)
            else:
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)




class Lenet_F(nn.Module):

    def __init__(self, K=10):
        super(Lenet_F, self).__init__()
        self.K = K 
        self.conv1 = nn.Conv2d(1, 6, 5,stride=1,padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(400, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3_bayes   = nn.Linear(84, self.K * self.K)


    def forward(self, x):

        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3_bayes(out) # (N, K*K)
        out = out.reshape(out.size(0), self.K, self.K) # (n, K, K)

        prob = F.softmax(out, 2)
        return prob
    
    def get_feature(self, x):

        output_list = []

        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        output_list.append(out)

        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        output_list.append(out)

        feature_num = len(output_list)
        for i in range(feature_num):
            output_list[i] = output_list[i].reshape(output_list[i].shape[0], -1)

        return output_list
    
    def copy_structure(self):
        return Lenet()
                        
  