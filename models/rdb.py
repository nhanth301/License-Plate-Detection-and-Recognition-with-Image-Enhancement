import torch
import torch.nn as nn

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