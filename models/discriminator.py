# file: models/discriminator.py
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()

        layers = []
        in_ch = in_channels
        
        for feature in features:
            layers.append(
                spectral_norm(
                    nn.Conv2d(
                        in_channels=in_ch,
                        out_channels=feature,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False, 
                    )
                )
            )

            if in_ch != in_channels:
                layers.append(nn.BatchNorm2d(feature))
            
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = feature

        layers.append(
            spectral_norm(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=1,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                )
            )
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)