import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from typing import List
Tensor = torch.Tensor



class ResnetBlock(nn.Module):
    """
    Args:
        dim (int): The number of channels for the input and output tensors.
                   The channel dimension remains constant throughout the block.
    """
    def __init__(self, dim: int):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the forward pass for the ResNet block.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor, which is the sum of the input tensor
                    and the output of the convolutional block (the skip-connection).
        """
        return x + self.conv_block(x)

class Generator(nn.Module):
    """
    Defines the ResNet-based Generator network.

    The architecture follows a standard encoder-transformer-decoder structure:
    1. An initial convolutional layer.
    2. A series of downsampling layers (the encoder).
    3. A series of ResNet blocks for feature transformation (the transformer).
    4. A series of upsampling layers (the decoder).
    5. A final output layer that projects back to the image space.

    Args:
        in_channels (int): Number of channels in the input image (e.g., 3 for RGB).
        out_channels (int): Number of channels in the output image.
        n_resnet_blocks (int): The number of ResNet blocks to use in the transformer part.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 3, n_resnet_blocks: int = 9):
        super().__init__()

        model: List[nn.Module] = []

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        ]

        in_features = 64
        out_features = 128
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(True)
            ]
            in_features = out_features
            out_features *= 2 

        for _ in range(n_resnet_blocks):
            model += [ResnetBlock(in_features)]

        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_features, out_features, kernel_size=3, padding=1, bias=True),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(True)
            ]
            in_features = out_features
            out_features //= 2

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class Discriminator(nn.Module):
    """
    Defines a PatchGAN-style Discriminator with spectral normalization.
    Args:
        in_channels (int): Number of channels in the input image.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()

        def _discriminator_block(in_filters: int, out_filters: int) -> List[nn.Module]:
            """Returns a Conv-InstanceNorm-LeakyReLU block with spectral norm."""
            return [
                spectral_norm(nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1, bias=False)),
                nn.InstanceNorm2d(out_filters),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            *_discriminator_block(64, 128),
            *_discriminator_block(128, 256),

            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False)),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)