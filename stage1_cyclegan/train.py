# file: train.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os 
import argparse
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from models import Generator, Discriminator
from utils import ImagePool
from ..models.clear_generator import LPSR

# Định nghĩa Dataset
class ImageDataset(Dataset):
    def __init__(self, root, image_size=(64, 128)):
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.files_A = sorted([os.path.join(root, 'trainA', name) for name in os.listdir(os.path.join(root, 'trainA'))])
        self.files_B = sorted([os.path.join(root, 'trainB', name) for name in os.listdir(os.path.join(root, 'trainB'))])

    def __getitem__(self, index):
        img_A = Image.open(self.files_A[index % len(self.files_A)]).convert('RGB')
        img_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB')
        
        return self.transform(img_A), self.transform(img_B)

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netG_AtoB = Generator().to(device)
    netG_BtoA = LPSR(
        num_channels=3,
        num_features=32,
        growth_rate=16,
        num_blocks=4,
        num_layers=4,
        scale_factor=None
    ).to(device)
    netD_A = Discriminator().to(device)
    netD_B = Discriminator().to(device)

    netG_AtoB.apply(weights_init)
    netG_BtoA.apply(weights_init)
    netD_A.apply(weights_init)
    netD_B.apply(weights_init)

    # Loss functions
    criterion_GAN = torch.nn.MSELoss() # LSGAN loss
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Optimizers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_AtoB.parameters(), netG_BtoA.parameters()), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=args.lr, betas=(0.5, 0.999))


    # Dataloader
    dataloader = DataLoader(ImageDataset(args.dataroot, image_size=(args.height, args.width)), 
                            batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Buffers
    fake_A_pool = ImagePool(50)
    fake_B_pool = ImagePool(50)

    for epoch in range(args.epochs):
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{args.epochs}")
        for i, (real_A, real_B) in progress_bar:
            real_A, real_B = real_A.to(device), real_B.to(device)
            
            # --- Generators ---
            optimizer_G.zero_grad()

            # Identity loss
            # G_BtoA(A) should be A
            identity_A = netG_BtoA(real_A)
            loss_identity_A = criterion_identity(identity_A, real_A) * 5.0
            # G_AtoB(B) should be B
            identity_B = netG_AtoB(real_B)
            loss_identity_B = criterion_identity(identity_B, real_B) * 5.0

            # GAN loss
            fake_B = netG_AtoB(real_A)
            pred_fake = netD_B(fake_B)
            target_real = torch.ones_like(pred_fake, requires_grad=False)
            loss_GAN_AtoB = criterion_GAN(pred_fake, target_real)

            fake_A = netG_BtoA(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_BtoA = criterion_GAN(pred_fake, target_real)

            # Cycle loss
            recons_A = netG_BtoA(fake_B)
            loss_cycle_A = criterion_cycle(recons_A, real_A) * 10.0

            recons_B = netG_AtoB(fake_A)
            loss_cycle_B = criterion_cycle(recons_B, real_B) * 10.0

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_AtoB + loss_GAN_BtoA + loss_cycle_A + loss_cycle_B
            loss_G.backward()
            optimizer_G.step()

            # --- Discriminator A ---
            optimizer_D_A.zero_grad()
            
            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A = fake_A_pool.query(fake_A.detach())
            pred_fake = netD_A(fake_A)
            target_fake = torch.zeros_like(pred_fake, requires_grad=False)
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()
            optimizer_D_A.step()

            # --- Discriminator B ---
            optimizer_D_B.zero_grad()

            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)
            
            fake_B = fake_B_pool.query(fake_B.detach())
            pred_fake = netD_B(fake_B)
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()
            optimizer_D_B.step()
            
            progress_bar.set_postfix(G_loss=f"{loss_G.item():.4f}", D_loss=f"{(loss_D_A + loss_D_B).item():.4f}")

        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        torch.save(netG_AtoB.state_dict(), f'checkpoints/netG_AtoB_epoch_{epoch}.pth')
        torch.save(netG_BtoA.state_dict(), f'checkpoints/netG_BtoA_epoch_{epoch}.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--dataroot", type=str, required=True, help="Path to the dataset root directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--height", type=int, default=64, help="Image height")
    parser.add_argument("--width", type=int, default=128, help="Image width")
    args = parser.parse_args()
    train(args)