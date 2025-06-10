from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms

class MyDataset(Dataset):
    def __init__(self, clear_img_path, blur_img_path, image_size=(64, 128)):
        self.clear_img_path = clear_img_path
        self.blur_img_path = blur_img_path
        self.image_size = image_size

        self.clear_images = sorted([
            os.path.join(clear_img_path, f)
            for f in os.listdir(clear_img_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.blur_images = sorted([
            os.path.join(blur_img_path, f)
            for f in os.listdir(blur_img_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.total = len(self.clear_images) * len(self.blur_images)

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        clear_idx = idx // len(self.blur_images)
        blur_idx = idx % len(self.blur_images)

        clear_image = Image.open(self.clear_images[clear_idx]).convert("RGB")
        blur_image = Image.open(self.blur_images[blur_idx]).convert("RGB")

        clear_tensor = self.transform(clear_image)
        blur_tensor = self.transform(blur_image)

        return clear_tensor, blur_tensor
