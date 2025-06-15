# file: models.py
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

import torch
import torch.nn as nn

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_resnet_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(in_channels, 64, kernel_size=7, padding=0, bias=True),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(True) ]

        # Downsampling
        in_features = 64
        out_features = 128
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1, bias=True),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(True) ]
            in_features = out_features
            out_features *= 2

        # Residual blocks
        for _ in range(n_resnet_blocks):
            model += [ResnetBlock(in_features)]

        out_features = in_features // 2
        for _ in range(2):
            model += [  
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        nn.Conv2d(in_features, out_features, kernel_size=3, padding=1, bias=True),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(True) ]
            in_features = out_features
            out_features //= 2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, out_channels, kernel_size=7, padding=0),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, stride=1, padding=1, bias=False),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, stride=1, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x) 