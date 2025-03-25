import torch
import torch.nn as nn
import math
import torch.functional as F

class DenseLayer(nn.Module):
    def __init__(self, in_channelss, out_channels, bias=True):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channelss, out_channels, kernel_size=(3, 3), padding="same", bias=bias)
        self.ReLU = nn.ReLU(inplace=True)


    def forward(self, x):
        return torch.cat([x, self.ReLU(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channelss, growth_rate, num_layers, bias=True):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channelss + growth_rate * i, growth_rate, bias=bias) for i in range(num_layers)])
        self.lff = nn.Conv2d(in_channelss + growth_rate * num_layers, in_channelss, kernel_size=1, bias=bias)
        self.alpha = nn.Parameter(torch.tensor(1.0))
    def forward(self, x):
        return x + self.alpha * self.lff(self.layers(x))

class RDN(nn.Module):
    def __init__(self, num_channels, num_features, growth_rate, num_blocks, num_layers, bias=True):
        super(RDN, self).__init__()
        self.num_blocks = num_blocks

        self.shallowF1 = nn.Conv2d(num_channels, num_features, kernel_size=7, padding="same", bias=bias)
        self.shallowF2 = nn.Conv2d(num_features, num_features, kernel_size=7, padding="same", bias=bias)
        self.rdbs = nn.ModuleList()
        for _ in range(num_blocks):
            self.rdbs.append(RDB(num_features, growth_rate, num_layers))

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

class ChannelAttention(nn.Module):
    def __init__(self, channels_in):
        super(ChannelAttention, self).__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels_in, channels_in // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels_in // 4, channels_in),
            nn.Unflatten(1, (channels_in, 1, 1)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.block(x)
        return x * out

class UpScaling(nn.Module):
    def __init__(self, channels_in, scale_factor):
        super(UpScaling, self).__init__()
        num_stages = int(math.log2(scale_factor))
        layers = []
        for _ in range(num_stages):
            layers.append(
                nn.Conv2d(
                    in_channels=channels_in,
                    out_channels=channels_in * 4,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.PixelShuffle(upscale_factor=2))

        self.upscale = nn.Sequential(*layers)

    def forward(self, x):
        return self.upscale(x)

class LPSR(nn.Module):
    def __init__(self, num_channels, num_features, growth_rate, num_blocks, num_layers, scale_factor):
        super(LPSR, self).__init__()
        self.in_conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            padding=1,
        )
        self.rdn = RDN(num_channels, num_features, growth_rate, num_blocks, num_layers)
        self.ca = ChannelAttention(num_features)
        self.upscale = UpScaling(num_features, scale_factor)
        self.final_conv = nn.Conv2d(
            in_channels=num_features,
            out_channels=num_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        x = self.in_conv(x)
        x = self.rdn(x)
        x = self.ca(x)
        x = self.upscale(x)
        x = self.final_conv(x)
        return F.sigmoid(x)