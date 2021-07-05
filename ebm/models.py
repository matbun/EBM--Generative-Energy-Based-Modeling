import numpy as np

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_layers(m):
    if type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif type(m) == nn.Linear:
        m.weight.data.normal_(0, 0.01)
        m.bias.data.normal_(0, 0.01)
    
def sq_activation(x):
    return 0.5 * torch.pow(x,2)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class LeNet(nn.Module):
    """ Adapted LeNet
    - Swish activ. func.
    - padding=2 in first convo layer (instead of 0)
    """
    def __init__(self, out_dim=1, **kwargs):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2), #(28x28)
            nn.SiLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0), #(14x14)
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0), #(10x10)
            nn.SiLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0), #(5x5)
            nn.Flatten(),
            nn.Linear(5*5*16, 64),
            nn.SiLU(),
            nn.Linear(64, out_dim)
        )
        self.cnn_layers.apply(init_layers)

    def forward(self, x):
        o = self.cnn_layers(x).squeeze(dim=-1)
        return sq_activation(o)
    
    
class CNNModel(nn.Module):
    def __init__(self, hidden_features=32, out_dim=1, beta=0, gamma=0, **kwargs):
        """CNNModel
        beta: quadratic energies weigth. If not specified, the default is 0. If None, learnable parameter.
        gamma: Langevin "weight decay". If not specified, the default is 0. If None, learnable parameter.
        """
        super().__init__()
        # We increase the hidden dimension over layers. Here pre-calculated for simplicity.
        c_hid1 = hidden_features // 2
        c_hid2 = hidden_features
        c_hid3 = hidden_features * 2

        # Series of convolutions and Swish activation functions
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, c_hid1, kernel_size=5, stride=2,
                      padding=4),  # [16x16]
            Swish(),
            nn.Conv2d(c_hid1, c_hid2, kernel_size=3, stride=2,
                      padding=1),  #  [8x8]
            Swish(),
            nn.Conv2d(c_hid2, c_hid3, kernel_size=3, stride=2,
                      padding=1),  # [4x4]
            Swish(),
            nn.Conv2d(c_hid3, c_hid3, kernel_size=3, stride=2,
                      padding=1),  # [2x2]
            Swish(),
            nn.Flatten(),
            nn.Linear(c_hid3 * 4, c_hid3),

            Swish(),
            nn.Linear(c_hid3, out_dim)
        )
        self.cnn_layers.apply(init_layers)

    def forward(self, x):
        o = self.cnn_layers(x).squeeze(dim=-1)
        return sq_activation(o)


##############################################################
### DenseNet #################################################
##############################################################

import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import sys
import math

class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
#         self.bn1 = nn.BatchNorm2d(nChannels, affine=False)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
#         self.bn2 = nn.BatchNorm2d(interChannels, affine=False)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
#         out = self.conv1(F.relu(self.bn1(x)))
#         out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv1(F.relu(x))
        out = self.conv2(F.relu(out))
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
#         self.bn1 = nn.BatchNorm2d(nChannels, affine=False)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
#         out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv1(F.relu(x))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
#         self.bn1 = nn.BatchNorm2d(nChannels, affine=False)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(x)) #self.bn1(x)
        out = F.avg_pool2d(out, 2)
        return out

class DenseNet(nn.Module):
    """For EBM on MNIST: no batchnorm, input is 1 x 28 x 28"""
    def __init__(self, growthRate, reduction, nClasses=1, bottleneck=True, depth=11, gamma=0, activ_type=None, **kwargs):
        super(DenseNet, self).__init__()
        
        def erf_activ(x):
            return torch.erf(x)*np.sqrt(np.pi)/2 
        
        # Trash
        self.cnn_layers = []
        self.beta = torch.tensor(0)
        self.gamma = torch.tensor(gamma)
        self.activ = erf_activ if activ_type=="erf" else None
        if activ_type is not None:
            print("CNN: Initial activation:", activ_type)
        if gamma > 0:
            print("CNN: Using penalty")
            
        # Parabola params
        self.a = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.c = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        
        # Good...
        nDenseBlocks = 4
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(1, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

#         self.bn1 = nn.BatchNorm2d(nChannels, affine=False)
        self.fc = nn.Linear(nChannels, nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
#         if self.activ is not None:
#             x1 = self.activ(x) 
#         else:
#             x1=x
        x1 = x #torch.atan(x)
        out = self.conv1(x1)
        # print("conv", out.shape)
        out = self.trans1(self.dense1(out))
        # print("D1 + T1", out.shape)
        out = self.trans2(self.dense2(out))
        # print("D2 + T2", out.shape)
        out = self.dense3(out)
        # print("D3", out.shape)
        out = torch.squeeze(F.avg_pool2d(F.relu(out), 7)) #self.bn1(out)
        # print("Final", out.shape)
        # out = F.log_softmax(self.fc(out))
        out = self.fc(out)
        # print("FC", out.shape)
        
        # Penalize pixels outside [-1, 1]: 
        # Attenzione: quando voglio aumentare l'energia il modello potrebbe
        # cercare intenzionalmente di buttare i pixel fuori da questo intervallo!!
        # Rischio instabilità
        #penalty = self.gamma * (torch.exp(torch.pow(x, 4) / 4) - 1).sum() if self.gamma.item() > 0 else torch.tensor(0)
        
        #parabola = .5*torch.pow(self.a, 2) * out**2 + self.b * out + self.c #(self.b**2 / (4 * self.a))
        
        return -0.5*torch.pow(out, 2) #+ out #parabola #+ penalty
    