import torch
import torch.nn as nn


class DConv(nn.Module):
    def __init__(self, in_channels, out_channel, kernel_size, stride=1, padding='same', bias=False):
        super(DConv, self).__init__()

        self.dConv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, groups=in_channels, padding=padding, bias=bias),
            nn.Conv2d(in_channels, out_channel, kernel_size=1)
            )

    def forward(self, x):
        x = self.dConv(x)

        return x

class AutoEncoder(nn.Module):
    def __init__(self, in_channels, out_channel, expansion=4):
        super(AutoEncoder, self).__init__()

        self.conv_in = nn.Conv2d(in_channels, expansion*in_channels, kernel_size=7, stride=1, padding='same', bias=False)
        self.encoder = nn.Sequential(
            DConv(expansion*in_channels, expansion*in_channels, kernel_size=7),
            nn.PixelUnshuffle(2),
            nn.ReLU(inplace=True),
            DConv(expansion*in_channels*2**2, expansion*in_channels, kernel_size=7),
            nn.PixelUnshuffle(2),
            nn.ReLU(inplace=True),
            )

        self.decoder = nn.Sequential(
            DConv(expansion*in_channels*2**2, expansion*in_channels*2**2, kernel_size=7),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            DConv(expansion*in_channels, expansion*in_channels*2**2, kernel_size=7),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            )

        self.GA = nn.Sequential(self.encoder,
                                self.decoder,
                                )

        self.conv_out = nn.Conv2d(expansion*in_channels, out_channel, kernel_size=3, stride=1, padding='same', bias=False)

    def forward(self, x):
        _, _, h, w = x.size()
        # Check if height is not divisible by 16 (2^2)
        if h % 4 != 0:
            x = nn.functional.pad(x, (0, 0, 0, 4 - h % 4))

        # Check if width is not divisible by 16 (2^2)
        if w % 4 != 0:
            x = nn.functional.pad(x, (0, 4 - w % 4, 0, 0))

        conv_in = self.conv_in(x)
        out = self.GA(conv_in)
        out = conv_in + out

        out = self.conv_out(out)

        return out