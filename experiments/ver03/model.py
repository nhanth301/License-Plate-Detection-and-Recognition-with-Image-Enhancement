import torch
import torch.nn as nn
import math
import torch.functional as F

class IFENet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IFENet, self).__init__()
        self.ife = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.ife(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.block(x)

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
        
class FMM(nn.Module):
    def __init__(self, in_channels, out_channels, first_channels, num_layers):
        super(FMM, self).__init__()
        self.csar = nn.ModuleList([CSAR(in_channels=in_channels) for _ in range(num_layers)])
        self.conv = nn.Conv2d(in_channels=in_channels*2 + first_channels, out_channels=out_channels, kernel_size=3, padding=1)
    
    def forward(self, x, x_first):
        x_out = x
        for i in range(len(self.csar)):
            x_out = self.csar[i](x_out) 

        x_out = torch.cat([x, x_out, x_first], dim=1)
        x_out = self.conv(x_out)
        return x_out


class FTNet(nn.Module):
    def __init__(self, in_channels, fmm_in_channels, out_channels, num_blocks, num_layers):
        super(FTNet, self).__init__()
        self.conv_in = nn.Conv2d(in_channels=in_channels, out_channels=fmm_in_channels, kernel_size=3, padding=1)
        self.fmm_blocks = nn.ModuleList([FMM(in_channels=fmm_in_channels, out_channels=fmm_in_channels*2, first_channels=fmm_in_channels, num_layers=num_layers)])
        for i in range(1,num_blocks):
            self.fmm_blocks.append(FMM(in_channels=fmm_in_channels+fmm_in_channels*i, out_channels=fmm_in_channels+fmm_in_channels*(i+1), first_channels=fmm_in_channels, num_layers=num_layers))
        self.conv_out = nn.Conv2d(in_channels=fmm_in_channels+fmm_in_channels*num_blocks, out_channels=out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x_in = self.conv_in(x)
        x_out = self.fmm_blocks[0](x_in, x_in)
        for i in range(1,len(self.fmm_blocks)):
            x_out = self.fmm_blocks[i](x_out, x_in)
        x_out = self.conv_out(x_out)   
        return x_out + x

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

class CSFMNet(nn.Module):
    def __init__(self, in_channels, out_channels, fmm_channels, num_blocks, num_layers,scale_factor):
        super(CSFMNet,self).__init__()
        self.ife = IFENet(in_channels=in_channels, out_channels=in_channels)
        self.ftnet = FTNet(in_channels=in_channels, fmm_in_channels=fmm_channels, out_channels=out_channels, num_blocks=num_blocks,num_layers=num_layers)
        self.upscale = UpScaling(channels_in=out_channels, scale_factor=scale_factor)
        self.conv_out = nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.ife(x)
        x = self.ftnet(x)
        x = self.upscale(x)
        x = self.conv_out(x)
        return F.sigmoid(x)