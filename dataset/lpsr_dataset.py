import os

from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, transform):
        super().__init__()
        self.hr_dir = hr_dir
        self.lr_images = [
            (os.path.join(lr_dir, img), img) for img in os.listdir(lr_dir)
        ]
        self.transform = transform

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_image_path, lr_image_name = self.lr_images[idx]
        lr_image = Image.open(lr_image_path).convert("RGB")

        # Construct the corresponding HR image path
        # hr_image_name = lr_image_name.split(".png")[0]
        hr_image_path = os.path.join(self.hr_dir, lr_image_name)
        hr_image = Image.open(hr_image_path).convert("L")

        # Apply the correct transform to each image
        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return lr_image, hr_image