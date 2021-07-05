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
    

    
#################################
# ## TOY NETWORK FOR 2D DATA ## #
#################################

class ToyNet(nn.Module):
    def __init__(self, dim=2, n_f=32, leak=0.05):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(dim, n_f, 1, 1, 0),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f, n_f * 2, 1, 1, 0),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f * 2, n_f * 2, 1, 1, 0),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f * 2, n_f * 2, 1, 1, 0),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f * 2, 1, 1, 1, 0))
         
        #self.cnn_layers.apply(init_layers)
        
    def forward(self, x):
        e = self.cnn_layers(x).squeeze()
        return 0.5 * e**2