import torch
import torch.nn as nn
from models.rdb import RDB
from models.auto_encoder import AutoEncoder
from models.rdn import RDN
from models.csar import CSAR

class LPSR(nn.Module):
    def __init__(self, num_channels, num_features, growth_rate, num_blocks, num_layers, scale_factor):
        super(LPSR, self).__init__()
        self.auto_encoder = AutoEncoder(num_channels, num_channels)
        self.rdn = RDN(num_channels, num_features, growth_rate, num_blocks, num_layers)
        self.final_conv = nn.Conv2d(
            in_channels=num_features,
            out_channels=1,
            kernel_size=3,
            padding=1,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.auto_encoder(x)
        x = self.rdn(x)
        x = self.final_conv(x)
        return self.sigmoid(x)