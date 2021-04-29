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


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class LeNet(nn.Module):
    """ Adapted LeNet
    - Swish activ. func.
    - padding=0 in first convo layer (instead of 0)
    """
    def __init__(self, out_dim=1, **kwargs):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2), #(28x28)
            Swish(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0), #(14x14)
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0), #(10x10)
            Swish(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0), #(5x5)
            nn.Flatten(),
            nn.Linear(5*5*16, 64),
            Swish(),
            nn.Linear(64, out_dim)
        )
        self.cnn_layers.apply(init_layers)
        #self.beta = nn.Parameter(torch.rand(1))

    def forward(self, x):
        x = self.cnn_layers(x).squeeze(dim=-1)
        #TO-DO: add beta...
        return x
    
class CNNModel(nn.Module):
    """ Batchnorm makes the CDiv loss explode (in negative numbers)"""
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
            #nn.ReLU(),
            nn.Conv2d(c_hid1, c_hid2, kernel_size=3, stride=2,
                      padding=1),  #  [8x8]
            Swish(),
            #nn.ReLU(),
            nn.Conv2d(c_hid2, c_hid3, kernel_size=3, stride=2,
                      padding=1),  # [4x4]
            Swish(),
            #nn.ReLU(),
            nn.Conv2d(c_hid3, c_hid3, kernel_size=3, stride=2,
                      padding=1),  # [2x2]
            Swish(),
            #nn.ReLU(),
            nn.Flatten(),
            nn.Linear(c_hid3 * 4, c_hid3),

            Swish(),
            #nn.ReLU(),
            nn.Linear(c_hid3, out_dim)
        )
        self.cnn_layers.apply(init_layers)
        self.beta = nn.Parameter(torch.rand(1)) if beta is None else beta
        self.gamma = nn.Parameter(torch.rand(1)) if gamma is None else gamma

    def forward(self, x):
        x = self.cnn_layers(x).squeeze(dim=-1)
        return x