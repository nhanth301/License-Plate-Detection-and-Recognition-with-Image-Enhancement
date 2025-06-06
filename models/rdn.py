import torch
import torch.nn as nn
from models.rdb import RDB
from models.csar import CSAR

class RDN(nn.Module):
    def __init__(self, num_channels, num_features, growth_rate, num_blocks, num_layers, bias=True):
        super(RDN, self).__init__()
        self.num_blocks = num_blocks

        self.shallowF1 = nn.Conv2d(num_channels, num_features, kernel_size=7, padding="same", bias=bias)
        self.shallowF2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding="same", bias=bias)
        self.csar = CSAR(num_features)
        self.rdbs = nn.ModuleList()
        for _ in range(num_blocks):
            self.rdbs.append(RDB(num_features, growth_rate, num_layers))
            self.rdbs.append(self.csar)

        self.gff = nn.Sequential(
            nn.Conv2d(num_features*num_blocks, num_features, kernel_size=1, bias=bias),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding="same", bias=bias)
            )

    def forward(self, x):
        sfe1 = self.shallowF1(x)
        sfe2 = self.shallowF2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.num_blocks):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1

        return x