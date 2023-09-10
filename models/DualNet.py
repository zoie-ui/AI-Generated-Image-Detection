from models.noise.noise_3down import *
from models.rgb.rgb_3down import *
from module.crosstrans2 import *
import torch.nn as nn


class DualNet(nn.Module):
    def __init__(self):
        super(DualNet, self).__init__()
        self.noise = Trans_Noise()
        self.rgb = Pre2()
        self.ct = crosstrans(depth=2, dim=256, hidden_dim=1024, heads=4, head_dim=64, dropout=0.1)
        self.n_elayers = nn.Sequential(
            residual(256, 256),
            residual(256, 256)
        )
        self.r_elayers = nn.Sequential(
            residual(256, 256),
            residual(256, 256)
        )
        self.fc1 = nn.Linear(512, 2)

    def forward(self, x):
        b, _, h, w = x.size()
        n = self.noise(x)
        r = self.rgb(x)
        n, r = self.ct(n, r)
        #noise = torch.cat((noise,n), dim=1)
        #rgb = torch.cat((rgb, r), dim=1)
        n = self.n_elayers(n)
        r = self.r_elayers(r)
        x = torch.cat((n, r), dim=1)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size()[0], -1)
        x = self.fc1(x)
        return x
