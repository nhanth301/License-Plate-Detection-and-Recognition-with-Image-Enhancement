import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, in_channels),
            nn.Unflatten(1, (in_channels, 1, 1)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.block(x)
        return x * out
    
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels*2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.block(x)

class CSAR(nn.Module):
    def __init__(self, in_channels):
        super(CSAR, self).__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
        )
        self.ca = ChannelAttention(in_channels=in_channels)
        self.sa = SpatialAttention(in_channels=in_channels)
        self.conv_out = nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels, kernel_size=1)

    def forward(self, x):
        x_in = self.conv_in(x)
        x_ca = self.ca(x_in)
        x_sa = self.sa(x_in)
        x_out = torch.cat([x_in*x_ca, x_in*x_sa], dim=1)
        x_out = self.conv_out(x_out)
        return x + x_out