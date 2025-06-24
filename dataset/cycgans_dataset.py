# Standard library imports
import os
import random
from typing import List, Tuple

# Third-party imports
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Assuming 'Tensor' is a torch.Tensor after transformation
Tensor = transforms.ToTensor

class ImageDataset(Dataset):
    """
    A PyTorch Dataset for unpaired image-to-image translation tasks.

    This class loads images from two different domains (A and B) located in
    subdirectories of a root folder. It assumes an unpaired setup, where for each
    image requested from domain A, a *random* image is selected from domain B.

    The expected directory structure is:
    - root/
      - trainA/
        - image1.jpg
        - ...
      - trainB/
        - image101.jpg
        - ...

    Args:
        root (str): The root directory containing 'trainA' and 'trainB' folders.
        image_size (Tuple[int, int]): The target (height, width) to resize images to.
                                      Defaults to (32, 192).
    """
    def __init__(self, root: str, image_size: Tuple[int, int] = (32, 192)):
        super().__init__()

        # The transformation pipeline is defined directly, as in the original code.
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        # Define paths to the two domains to improve readability.
        path_A = os.path.join(root, 'trainA')
        path_B = os.path.join(root, 'trainB')

        # List and sort all file paths in each domain directory.
        self.files_A: List[str] = sorted([os.path.join(path_A, name) for name in os.listdir(path_A)])
        self.files_B: List[str] = sorted([os.path.join(path_B, name) for name in os.listdir(path_B)])

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        Retrieves an image from domain A and a random image from domain B.

        Args:
            index (int): The index for the image from domain A.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the transformed (image_A, image_B).
        """
        # The modulo operator ensures the index wraps around if it's larger than
        # the number of files, a standard practice for this dataset type.
        image_A_path = self.files_A[index % len(self.files_A)]
        
        # A random image is chosen from domain B to create an unpaired dataset.
        random_index = random.randint(0, len(self.files_B) - 1)
        image_B_path = self.files_B[random_index]
        
        # Open images and ensure they are in RGB format.
        img_A = Image.open(image_A_path).convert('RGB')
        img_B = Image.open(image_B_path).convert('RGB')
        
        # Apply the same transformation to both images.
        return self.transform(img_A), self.transform(img_B)

    def __len__(self) -> int:
        """
        Returns the total size of the dataset, defined by the number of images
        in the larger of the two domains.
        """
        return max(len(self.files_A), len(self.files_B))