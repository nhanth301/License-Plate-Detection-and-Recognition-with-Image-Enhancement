import argparse
import itertools
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from dataset.cycgans_dataset import ImageDataset
from my_models.cycle_gans import Discriminator, Generator
from torch.utils.data import DataLoader
from tqdm import tqdm
from my_utils.utils import ImagePool
from my_models.lpsr import LPSR

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and hasattr(m, "weight"):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    netG_AtoB = Generator().to(device)
    # netG_BtoA = Generator().to(device)
    netG_BtoA= LPSR(num_channels=3, num_features=32, growth_rate=16, num_blocks=4, num_layers=4, scale_factor=None).to(device)
    netD_A = Discriminator().to(device)
    netD_B = Discriminator().to(device)

    netG_AtoB.apply(weights_init)
    netG_BtoA.apply(weights_init)
    netD_A.apply(weights_init)
    netD_B.apply(weights_init)

    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    optimizer_G = torch.optim.Adam(
        itertools.chain(netG_AtoB.parameters(), netG_BtoA.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )
    optimizer_D_A = torch.optim.Adam(
        netD_A.parameters(), lr=args.lr, betas=(0.5, 0.999)
    )
    optimizer_D_B = torch.optim.Adam(
        netD_B.parameters(), lr=args.lr, betas=(0.5, 0.999)
    )

    dataloader = DataLoader(
        ImageDataset(args.dataroot, image_size=(args.height, args.width)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    fake_A_pool = ImagePool(50)
    fake_B_pool = ImagePool(50)

    def denormalize(img_tensor):
        img_tensor = img_tensor * 0.5 + 0.5
        img_tensor = img_tensor.clamp(0, 1)
        return img_tensor.permute(1, 2, 0).numpy()

    for epoch in range(args.epochs):
        progress_bar = tqdm(
            enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{args.epochs}"
        )
        for i, (real_A, real_B) in progress_bar:
            real_A, real_B = real_A.to(device), real_B.to(device)
            lambda_GAN = 2.0

            optimizer_G.zero_grad()
            netD_A.requires_grad_(False)
            netD_B.requires_grad_(False)

            loss_identity_A = criterion_identity(netG_BtoA(real_A), real_A) * 5.0
            loss_identity_B = criterion_identity(netG_AtoB(real_B), real_B) * 5.0

            fake_B = netG_AtoB(real_A)
            pred_fake = netD_B(fake_B)
            target_real = torch.ones_like(pred_fake, requires_grad=False)
            loss_GAN_AtoB = criterion_GAN(pred_fake, target_real) * lambda_GAN

            fake_A = netG_BtoA(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_BtoA = criterion_GAN(pred_fake, target_real) * lambda_GAN

            recons_A = netG_BtoA(fake_B)
            loss_cycle_A = criterion_cycle(recons_A, real_A) * 10.0
            recons_B = netG_AtoB(fake_A)
            loss_cycle_B = criterion_cycle(recons_B, real_B) * 20.0

            loss_G = (
                loss_identity_A
                + loss_identity_B
                + loss_GAN_AtoB
                + loss_GAN_BtoA
                + loss_cycle_A
                + loss_cycle_B
            )
            loss_G.backward()
            optimizer_G.step()

            optimizer_D_A.zero_grad()
            netD_A.requires_grad_(True)

            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            fake_A_pooled = fake_A_pool.query(fake_A.detach())
            pred_fake = netD_A(fake_A_pooled)
            target_fake = torch.zeros_like(pred_fake, requires_grad=False)
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()
            optimizer_D_A.step()

            optimizer_D_B.zero_grad()
            netD_B.requires_grad_(True)

            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)

            fake_B_pooled = fake_B_pool.query(fake_B.detach())
            pred_fake = netD_B(fake_B_pooled)
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()
            optimizer_D_B.step()

            progress_bar.set_postfix(
                G_loss=f"{loss_G.item():.4f}",
                D_loss=f"{(loss_D_A + loss_D_B).item():.4f}",
            )

        with torch.no_grad():
            netG_AtoB.eval()
            netG_BtoA.eval()

            fake_B = netG_AtoB(real_A)
            recons_A = netG_BtoA(fake_B)
            fake_A = netG_BtoA(real_B)
            recons_B = netG_AtoB(fake_A)

            img_list = [
                denormalize(real_A[0].cpu()),
                denormalize(fake_B[0].cpu()),
                denormalize(recons_A[0].cpu()),
                denormalize(real_B[0].cpu()),
                denormalize(fake_A[0].cpu()),
                denormalize(recons_B[0].cpu()),
            ]
            titles = [
                "Real Clear (A)",
                "Fake Blur (B)",
                "Reconstructed Clear (A)",
                "Real Blur (B)",
                "Fake Clear (A)",
                "Reconstructed Blur (B)",
            ]

            fig, axs = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f"Epoch: {epoch+1}", fontsize=16)
            for i, ax in enumerate(axs.flat):
                ax.imshow(img_list[i])
                ax.set_title(titles[i])
                ax.axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f"epoch_{epoch+1}.png"))
            plt.close()

            netG_AtoB.train()
            netG_BtoA.train()

        if epoch % 50 == 0 or epoch == args.epochs - 1:
            torch.save(
                netG_AtoB.state_dict(),
                os.path.join(args.checkpoint_dir, f"netG_AtoB_epoch_{epoch+1}.pth"),
            )
            torch.save(
                netG_BtoA.state_dict(),
                os.path.join(args.checkpoint_dir, f"netG_BtoA_epoch_{epoch+1}.pth"),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument(
        "--dataroot",
        type=str,
        required=True,
        help="Path to the dataset root directory",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--height", type=int, default=24, help="Image height")
    parser.add_argument("--width", type=int, default=192, help="Image width")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_images",
        help="Directory to save visualization images",
    )

    args = parser.parse_args()
    train(args)