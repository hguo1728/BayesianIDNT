import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch.hub as hub
import torchvision.models as models
from torch.autograd import Variable
from torchvision import transforms

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# ---------------- ResNet: Base model ---------------- 

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.num_blocks = num_blocks
        self.num_classes = num_classes

        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.T_revision = nn.Linear(self.num_classes, self.num_classes, False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, revision=False):

        correction = self.T_revision.weight

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        out = self.linear(out)
        prob = F.softmax(out, 1)

        if revision == True:
            return prob, correction
        else:
            return prob
    
    def get_feature(self, x):
        output_list = []

        out = F.relu(self.bn1(self.conv1(x)))

        for name, module in self.layer1._modules.items():
            out = module(out)
        for name, module in self.layer2._modules.items():
            out = module(out)
        for name, module in self.layer3._modules.items():
            out = module(out)
        for name, module in self.layer4._modules.items():
            out = module(out)
            output_list.append(out)

        feature_num = len(output_list)
        for i in range(feature_num):
            output_list[i] = output_list[i].reshape(output_list[i].shape[0], -1)

        return output_list
    
    def copy_structure(self):
        return ResNet(BasicBlock, self.num_blocks, self.num_classes)


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)
    
def ResNet34(num_classes):
    return ResNet(BasicBlock, [3,4,6,3], num_classes)



# ---------------- ResNet: Transition model ---------------- 

class ResNet_T(nn.Module):
    def __init__(self, block, num_blocks, num_classes, R):
        super(ResNet_T, self).__init__()
        self.num_blocks = num_blocks
        self.K = num_classes
        self.R = R

        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1_1 = nn.Linear(512*block.expansion, 512)
        self.fc1_2 = nn.Linear(512*block.expansion, 512)

        self.out_1 = nn.Linear(512, self.K * self.K)
        self.out_2 = nn.Linear(512, self.K * self.R)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        out1 = F.relu(self.fc1_1(out)) # 512*block.expansion -> 512
        out1 = self.out_1(out1) # 512 -> K * K
        out1 = out1.reshape(out1.size(0), self.K, self.K) # (n, K, K)

        out2 = F.relu(self.fc1_2(out)) # 512*block.expansion -> 512
        out2 = self.out_2(out2) # 512 -> R * K
        out2 = out2.reshape(out2.size(0), self.R, self.K) # (n, R, K)

        out = out1[:, None, :, :] + out2[:, :, None, :]
        prob = F.softmax(out, dim=3) # (n, R, K, K)

        return prob
    
    def get_pretrained_weights(self, model_base):
        model_base.eval()
        for name, module in self.named_children():
            if name == 'conv1':
                dict = model_base.conv1.state_dict()
                module.load_state_dict(dict)
            elif name == 'bn1':
                dict = model_base.bn1.state_dict()
                module.load_state_dict(dict)
            elif name == 'layer1':
                dict = model_base.layer1.state_dict()
                module.load_state_dict(dict)
            elif name == 'layer2':
                dict = model_base.layer2.state_dict()
                module.load_state_dict(dict)
            elif name == 'layer3':
                dict = model_base.layer3.state_dict()
                module.load_state_dict(dict)
            elif name == 'layer4':
                dict = model_base.layer4.state_dict()
                module.load_state_dict(dict)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


def ResNet18_T(num_classes, R):
    return ResNet_T(BasicBlock, [2,2,2,2], num_classes, R)
    
def ResNet34_T(num_classes, R):
    return ResNet_T(BasicBlock, [3,4,6,3], num_classes, R)


# ---------------- ResNet: Bayes (BLTM) ---------------- 

class ResNet_F(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet_F, self).__init__()
        self.num_blocks = num_blocks
        self.K = num_classes

        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bayes_linear = nn.Linear(512 * block.expansion, self.K * self.K)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)

        out = out.view(out.size(0), -1) # (n, 512*block.expansion)
        out = self.bayes_linear(out) # 512*block.expansion -> K * K
        out = out.reshape(out.size(0), self.K, self.K) # (n, K, K)
        out = F.softmax(out, dim=2)

        return out

def ResNet18_F(num_classes):
    return ResNet_F(BasicBlock, [2,2,2,2], num_classes)

def ResNet34_F(num_classes):
    return ResNet_F(BasicBlock, [3,4,6,3], num_classes)
    
       
