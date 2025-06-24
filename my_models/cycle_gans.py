import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from typing import List
Tensor = torch.Tensor


# --- Model Components ---

class ResnetBlock(nn.Module):
    """
    Defines a single Residual Block, a key component of ResNet architectures.

    A ResNet block consists of two convolutional layers with a skip-connection
    that adds the input of the block to its output. This helps to prevent the
    vanishing gradient problem, allowing for much deeper networks.

    Args:
        dim (int): The number of channels for the input and output tensors.
                   The channel dimension remains constant throughout the block.
    """
    def __init__(self, dim: int):
        # Use the modern super() call, which is standard in Python 3.
        super().__init__()

        # The main convolutional path is defined as a sequential block.
        self.conv_block = nn.Sequential(
            # Reflection padding is used instead of zero-padding to reduce
            # potential border artifacts in image generation tasks.
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
        # The skip-connection is the core idea of a ResNet block.
        return x + self.conv_block(x)


# --- Main GAN Models ---

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

        # The model is constructed as a list of layers, which will be passed to
        # nn.Sequential. This is a common pattern for building sequential models.
        model: List[nn.Module] = []

        # --- 1. Initial Convolution Block ---
        # This block serves as the entry point, mapping the input image to a
        # higher-dimensional feature space. A large kernel is used to capture
        # a wide receptive field.
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        ]

        # --- 2. Downsampling Blocks (Encoder) ---
        # Two downsampling blocks are used to reduce the spatial dimensions of
        # the feature map, which helps the network learn more abstract features.
        in_features = 64
        out_features = 128
        for _ in range(2):
            # A strided convolution (stride=2) halves the height and width.
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(True)
            ]
            in_features = out_features
            out_features *= 2 # Double the number of channels for the next block.

        # --- 3. Residual Blocks (Transformer) ---
        # This is the core of the generator, where the main feature transformation occurs.
        # The ResNet blocks do not change the spatial dimensions of the feature map.
        for _ in range(n_resnet_blocks):
            model += [ResnetBlock(in_features)]

        # --- 4. Upsampling Blocks (Decoder) ---
        # Two upsampling blocks are used to restore the original spatial dimensions.
        out_features = in_features // 2
        for _ in range(2):
            # Upsample followed by a convolution is a common way to increase resolution
            # while avoiding the checkerboard artifacts that can be caused by
            # transposed convolutions.
            model += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_features, out_features, kernel_size=3, padding=1, bias=True),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(True)
            ]
            in_features = out_features
            out_features //= 2 # Halve the number of channels.

        # --- 5. Output Layer ---
        # This final block maps the features back to the desired output image format.
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, kernel_size=7, padding=0),
            # Tanh activation scales the output pixel values to the range [-1, 1],
            # a common convention for GAN-generated images.
            nn.Tanh()
        ]

        # Unpack the list of layers into a sequential model.
        self.model = nn.Sequential(*model)

    def forward(self, x: Tensor) -> Tensor:
        """Defines the forward pass through the generator network."""
        return self.model(x)


class Discriminator(nn.Module):
    """
    Defines a PatchGAN-style Discriminator with spectral normalization.

    Instead of classifying the entire image as real or fake, a PatchGAN
    discriminator outputs a feature map where each element corresponds to the
    "realness" of a patch in the input image. This encourages the generator
    to produce realistic local details. Spectral normalization is applied to
    the convolutional layers to stabilize GAN training.

    Args:
        in_channels (int): Number of channels in the input image.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()

        # This nested helper function creates a standard discriminator block.
        # Using a helper makes the main architecture definition cleaner and
        # avoids repeating the same sequence of layers (DRY principle).
        def _discriminator_block(in_filters: int, out_filters: int) -> List[nn.Module]:
            """Returns a Conv-InstanceNorm-LeakyReLU block with spectral norm."""
            # bias is set to False as InstanceNorm includes a learnable bias.
            return [
                spectral_norm(nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1, bias=False)),
                nn.InstanceNorm2d(out_filters),
                # LeakyReLU is used to prevent the "dying ReLU" problem.
                nn.LeakyReLU(0.2, inplace=True)
            ]

        # The full model is built as a single sequential chain.
        self.model = nn.Sequential(
            # --- Initial Convolution Block ---
            # This block doesn't use InstanceNorm, a common choice for the first layer.
            spectral_norm(nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            # --- Downsampling Blocks ---
            # Unpack the list of layers returned by the helper function.
            *_discriminator_block(64, 128),
            *_discriminator_block(128, 256),

            # --- Middle Block ---
            # This block uses a stride of 1, so it does not reduce the spatial
            # dimensions further. It is used for deeper feature extraction.
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False)),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # --- Final Output Layer ---
            # This final convolution maps the feature map to a single channel,
            # producing the N x N patch output. No activation function is applied here,
            # as the loss function (e.g., BCEWithLogitsLoss) will handle it.
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Defines the forward pass through the discriminator network."""
        return self.model(x)