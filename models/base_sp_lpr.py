import torch
import torch.nn as nn
import math

class DenseLayer(nn.Module):
    """
    A dense layer consisting of a convolution and a ReLU activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bias (bool, optional): Whether to use bias in the convolution. Defaults to True.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding="same", bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the dense layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return torch.cat([x, self.relu(self.conv(x))], 1)

class RDB(nn.Module):
    """
    Residual Dense Block (RDB).

    Args:
        in_channels (int): Number of input channels.
        growth_rate (int): Growth rate of the dense layers.
        num_layers (int): Number of dense layers in the RDB.
        bias (bool, optional): Whether to use bias in the convolutions. Defaults to True.
    """
    def __init__(self, in_channels: int, growth_rate: int, num_layers: int, bias: bool = True):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate, bias=bias) for i in range(num_layers)])
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, in_channels, kernel_size=1, bias=bias)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RDB.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return x + self.alpha * self.lff(self.layers(x))

class RDN(nn.Module):
    """
    Residual Dense Network (RDN).

    Args:
        num_channels (int): Number of input channels.
        num_features (int): Number of features in the shallow layers and RDBs.
        growth_rate (int): Growth rate of the dense layers in the RDBs.
        num_blocks (int): Number of RDBs.
        num_layers (int): Number of dense layers in each RDB.
        bias (bool, optional): Whether to use bias in the convolutions. Defaults to True.
    """
    def __init__(self, num_channels: int, num_features: int, growth_rate: int, num_blocks: int, num_layers: int, bias: bool = True):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RDN.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
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
    """
    Channel Attention module.

    Args:
        channels_in (int): Number of input channels.
    """
    def __init__(self, channels_in: int):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the channel attention module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = self.block(x)
        return x * out
    
class UpScaling(nn.Module):
    """
    Upscaling module using pixel shuffle.

    Args:
        channels_in (int): Number of input channels.
        scale_factor (int): Upscaling factor.
    """
    def __init__(self, channels_in: int, scale_factor: int):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the upscaling module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.upscale(x)

class LPSR(nn.Module):
    """
    License Plate Super-Resolution (LPSR) model.

    Args:
        num_channels (int): Number of input channels.
        num_features (int): Number of features in the shallow layers and RDBs.
        growth_rate (int): Growth rate of the dense layers in the RDBs.
        num_blocks (int): Number of RDBs.
        num_layers (int): Number of dense layers in each RDB.
        scale_factor (int): Upscaling factor.
    """
    def __init__(self, num_channels: int, num_features: int, growth_rate: int, num_blocks: int, num_layers: int, scale_factor: int):
        super(LPSR, self).__init__()
        self.rdn = RDN(num_channels, num_features, growth_rate, num_blocks, num_layers)
        self.ca = ChannelAttention(num_features)
        self.upscale = UpScaling(num_features, scale_factor)
        self.final_conv = nn.Conv2d(
            in_channels=num_features,
            out_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LPSR model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.rdn(x)
        x = self.ca(x)
        x = self.upscale(x)
        x = self.final_conv(x)
        return x
    
if __name__ == "__main__":
    model = LPSR(num_channels=3, num_features=64, growth_rate=64, num_blocks=16, num_layers=8, scale_factor=1)
    input = torch.randn(1, 3, 64, 64)   
    print("Input shape: ", input.shape)
    output = model(input)
    print("Output shape: ", output.shape)
