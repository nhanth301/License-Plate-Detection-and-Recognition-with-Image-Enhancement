import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class DConv(nn.Module):
    def __init__(
        self, in_channels, out_channel, kernel_size, stride=1, padding="same", bias=True
    ):
        super().__init__()
        self.dConv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                groups=in_channels,
                padding=padding,
                bias=bias,
            ),
            nn.Conv2d(in_channels, out_channel, kernel_size=1),
        )

    def forward(self, x):
        x = self.dConv(x)
        return x


class DenseLayer(nn.Module):
    def __init__(self, in_channelss, out_channels, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channelss, out_channels, kernel_size=(3, 3), padding="same", bias=bias
        )
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.ReLU(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channelss, growth_rate, num_layers, bias=True):
        super().__init__()
        self.layers = nn.Sequential(
            *[
                DenseLayer(in_channelss + growth_rate * i, growth_rate, bias=bias)
                for i in range(num_layers)
            ]
        )
        self.lff = nn.Conv2d(
            in_channelss + growth_rate * num_layers,
            in_channelss,
            kernel_size=1,
            bias=bias,
        )
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x + self.alpha * self.lff(self.layers(x))


class AutoEncoder(nn.Module):
    def __init__(self, in_channels, out_channel, expansion=4, kernel_size=5):
        super().__init__()
        self.conv_in = nn.Conv2d(
            in_channels, expansion * in_channels, kernel_size=3, stride=1, padding="same", bias=False
        )
        self.encoder = nn.Sequential(
            DConv(expansion * in_channels, expansion * in_channels, kernel_size=kernel_size),
            nn.PixelUnshuffle(2),
            nn.ReLU(inplace=True),
            DConv(
                expansion * in_channels * 2**2,
                expansion * in_channels,
                kernel_size=kernel_size,
            ),
            nn.PixelUnshuffle(2),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            DConv(
                expansion * in_channels * 2**2,
                expansion * in_channels * 2**2,
                kernel_size=kernel_size,
            ),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            DConv(
                expansion * in_channels,
                expansion * in_channels * 2**2,
                kernel_size=kernel_size,
            ),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )
        self.GA = nn.Sequential(
            self.encoder,
            self.decoder,
        )
        self.conv_out = nn.Conv2d(
            expansion * in_channels, out_channel, kernel_size=3, stride=1, padding="same", bias=False
        )

    def forward(self, x):
        _, _, h, w = x.size()
        if h % 4 != 0:
            x = nn.functional.pad(x, (0, 0, 0, 4 - h % 4))
        if w % 4 != 0:
            x = nn.functional.pad(x, (0, 4 - w % 4, 0, 0))

        conv_in = self.conv_in(x)
        out = self.GA(conv_in)
        out = conv_in + out
        out = self.conv_out(out)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
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
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=in_channels * 2, kernel_size=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels * 2, out_channels=in_channels, kernel_size=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.block(x)


class CSAR(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=1,
            ),
        )
        self.ca = ChannelAttention(in_channels=in_channels)
        self.sa = SpatialAttention(in_channels=in_channels)
        self.conv_out = nn.Conv2d(
            in_channels=in_channels * 2, out_channels=in_channels, kernel_size=1
        )

    def forward(self, x):
        x_in = self.conv_in(x)
        x_ca = self.ca(x_in)
        x_sa = self.sa(x_in)
        x_out = torch.cat([x_in * x_ca, x_in * x_sa], dim=1)
        x_out = self.conv_out(x_out)
        return x + x_out


class RDN(nn.Module):
    def __init__(
        self, num_channels, num_features, growth_rate, num_blocks, num_layers, bias=True
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.shallowF1 = nn.Conv2d(
            num_channels, num_features, kernel_size=7, padding="same", bias=bias
        )
        self.shallowF2 = nn.Conv2d(
            num_features, num_features, kernel_size=3, padding="same", bias=bias
        )
        self.csar = CSAR(num_features)
        self.rdbs = nn.ModuleList()
        for _ in range(num_blocks):
            self.rdbs.append(RDB(num_features, growth_rate, num_layers))
            self.rdbs.append(self.csar)

        self.gff = nn.Sequential(
            nn.Conv2d(
                num_features * num_blocks, num_features, kernel_size=1, bias=bias
            ),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding="same", bias=bias),
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


class UpScaling(nn.Module):
    def __init__(self, channels_in, scale_factor):
        super().__init__()
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
    def __init__(
        self, num_channels, num_features, growth_rate, num_blocks, num_layers, scale_factor
    ):
        super().__init__()
        self.auto_encoder = AutoEncoder(num_channels, num_channels)
        self.rdn = RDN(
            num_channels, num_features, growth_rate, num_blocks, num_layers
        )
        # self.upscale = UpScaling(num_features, scale_factor)
        self.final_conv = nn.Conv2d(
            in_channels=num_features,
            out_channels=1,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        x = self.auto_encoder(x)
        x = self.rdn(x)
        # x = self.upscale(x)
        x = self.final_conv(x)
        return torch.sigmoid(x)