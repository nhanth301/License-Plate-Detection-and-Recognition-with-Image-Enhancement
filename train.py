import torch
import torch.optim as optim
import torch.nn as nn
from dataset.my_dataset import MyDataset
from torch.utils.data import DataLoader
import os 
import argparse
import matplotlib.pyplot as plt
from torch.nn import functional as F
from models.clear_generator import LPSR
from models.blur_generator import BlurGenerator
from models.discriminator import Discriminator
from torchvision import models
import shutil
from tqdm import tqdm

def train_models(blur_generator, discriminator, clear_generator,
                 dataloader, num_epochs, device, save_path="models", vs_save_path="visualize"):

    if not os.path.exists(vs_save_path):
        os.makedirs(vs_save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()

    optimizer_blur_gen = optim.Adam(blur_generator.parameters(), lr=0.0002)
    optimizer_clear_gen = optim.Adam(clear_generator.parameters(), lr=0.0002)
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=0.0002)

    # Learning rate schedulers
    scheduler_blur_gen = optim.lr_scheduler.StepLR(optimizer_blur_gen, step_size=10, gamma=0.5)
    scheduler_clear_gen = optim.lr_scheduler.StepLR(optimizer_clear_gen, step_size=10, gamma=0.5)
    scheduler_disc = optim.lr_scheduler.StepLR(optimizer_disc, step_size=10, gamma=0.5)

    # Perceptual loss with VGG16
    vgg = models.vgg16(pretrained=True).features.to(device).eval()
    def perceptual_loss(img1, img2):
        vgg_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img1.device)
        vgg_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img1.device)
        img1_norm = (img1 - vgg_mean) / vgg_std
        img2_norm = (img2 - vgg_mean) / vgg_std
        feat1 = vgg(img1_norm)
        feat2 = vgg(img2_norm)
        return nn.MSELoss()(feat1, feat2)
    
    def calc_mean_std(feat, eps=1e-5):
        size = feat.size()
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def style_loss(fake_feats, real_feats):
        loss = 0.0
        for f_fake, f_real in zip(fake_feats, real_feats):
            mean_fake, std_fake = calc_mean_std(f_fake)
            mean_real, std_real = calc_mean_std(f_real)
            loss += F.l1_loss(mean_fake, mean_real) + F.l1_loss(std_fake, std_real)
        return loss
    
    def extract_features(x):
        features = []
        for i, layer in enumerate(vgg):
            x = layer(x)
            if i in {3, 8, 15}:  
                features.append(x)
        return features

    best_sr_loss = float('inf')

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        blur_generator.train()
        clear_generator.train()
        discriminator.train()

        total_gen_loss = 0
        total_disc_loss = 0
        total_sr_loss = 0
        for batch_idx, (clear_img, blur_img) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            clear_img, blur_img = clear_img.to(device), blur_img.to(device)
            batch_size = clear_img.size(0)

            # Generate fake blurred image
            fake_blur = blur_generator(clear_img, blur_img)

            # Train Discriminator
            real_labels = torch.full((batch_size, 1), 0.9, device=device)
            fake_labels = torch.zeros((batch_size, 1), device=device)
            real_input = blur_img + 0.05 * torch.randn_like(blur_img)
            fake_input = fake_blur.detach() + 0.05 * torch.randn_like(fake_blur)

            real_output = discriminator(real_input)
            fake_output = discriminator(fake_input)

            disc_real_loss = bce_loss(real_output, real_labels)
            disc_fake_loss = bce_loss(fake_output, fake_labels)
            disc_loss = (disc_real_loss + disc_fake_loss) / 2

            optimizer_disc.zero_grad()
            disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizer_disc.step()
            total_disc_loss += disc_loss.item()

            # Train Blur Generator and Clear Generator
            gen_labels = torch.full((batch_size, 1), 0.9, device=device)
            fake_output_for_gen = discriminator(fake_blur)

            gen_adv_loss = bce_loss(fake_output_for_gen, gen_labels)
            fake_feats = extract_features(fake_blur)
            real_feats = extract_features(blur_img)
            gen_style_loss = style_loss(fake_feats, real_feats)

            # Content loss using VGG features (layer 8)
            clear_feats = extract_features(clear_img)
            content_loss = nn.MSELoss()(fake_feats[1], clear_feats[1])

            # Clear generator output
            hr_output = clear_generator(fake_blur)

            sr_recon_loss = 5 * l1_loss(hr_output, clear_img) + l2_loss(hr_output, clear_img)
            sr_perc_loss = perceptual_loss(hr_output, clear_img)
            sr_loss = sr_recon_loss + 0.1 * sr_perc_loss

            # Total loss for blur generator
            total_blur_gen_loss = gen_adv_loss + 0.1 * gen_style_loss + 0.01 * content_loss + sr_loss

            # Zero gradients for both optimizers
            optimizer_blur_gen.zero_grad()
            optimizer_clear_gen.zero_grad()

            # Backward pass
            total_blur_gen_loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(blur_generator.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(clear_generator.parameters(), max_norm=1.0)

            # Update parameters
            optimizer_blur_gen.step()
            optimizer_clear_gen.step()

            total_gen_loss += (gen_adv_loss + 0.1 * gen_style_loss + 0.01 * content_loss).item()
            total_sr_loss += sr_loss.item()

        # Update schedulers
        scheduler_blur_gen.step()
        scheduler_clear_gen.step()
        scheduler_disc.step()

        # Calculate average losses
        avg_gen_loss = total_gen_loss / len(dataloader)
        avg_disc_loss = total_disc_loss / len(dataloader)
        avg_sr_loss = total_sr_loss / len(dataloader)

        print(f"Epoch {epoch+1}/{num_epochs}, Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}, SR Loss: {avg_sr_loss:.4f}")

        # Save models if SR loss improves
        if avg_sr_loss < best_sr_loss:
            best_sr_loss = avg_sr_loss
            torch.save(blur_generator.state_dict(), os.path.join(save_path, "best_blur_generator.pth"))
            torch.save(clear_generator.state_dict(), os.path.join(save_path, "best_clear_generator.pth"))
            torch.save(discriminator.state_dict(), os.path.join(save_path, "best_discriminator.pth"))
            print(f"Saved best models at epoch {epoch+1}")

        # Save periodic checkpoints
        if epoch % 20 == 0:
            torch.save(blur_generator.state_dict(), os.path.join(save_path, f"generator_{epoch}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(save_path, f"discriminator_{epoch}.pth"))
            torch.save(clear_generator.state_dict(), os.path.join(save_path, f"sr_{epoch}.pth"))

        # Visualize images (first sample of the last batch)
        clear_img_cpu = clear_img[0].detach().cpu()
        fake_blur_cpu = fake_blur[0].detach().cpu()
        real_blur_cpu = blur_img[0].detach().cpu()
        sr_img_cpu = hr_output[0].detach().cpu()

        def denormalize(img_tensor):
            img_tensor = img_tensor.clamp(0, 1)
            return img_tensor.permute(1, 2, 0).numpy()

        plt.figure(figsize=(16, 4))
        plt.subplot(1, 4, 1)
        plt.title("Clear Image (GT)")
        plt.imshow(denormalize(clear_img_cpu))
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.title("Fake Blur")
        plt.imshow(denormalize(fake_blur_cpu))
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.title("SR Image")
        plt.imshow(denormalize(sr_img_cpu))
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.title("Real Blur")
        plt.imshow(denormalize(real_blur_cpu))
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(f"{vs_save_path}/output_epoch_{epoch}.png")
        plt.close()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset and dataloader
    dataset = MyDataset(args.clear_folder, args.blur_folder, image_size=(64, 128))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    blur_generator = BlurGenerator(in_channels=3, out_channels=3, feature_dim=64).to(device)
    clear_generator = LPSR(num_channels=3,
                           num_features=32,
                           growth_rate=16,
                           num_blocks=4,
                           num_layers=4,
                           scale_factor=None).to(device)
    discriminator = Discriminator().to(device)

    # Train models
    train_models(blur_generator, discriminator, clear_generator,
                 dataloader, args.epochs, device, save_path=args.save_path, vs_save_path=args.vs_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image deblurring model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--clear_folder", type=str, help="Path to clear image folder")
    parser.add_argument("--blur_folder", type=str, help="Path to blurred image folder")
    parser.add_argument("--save_path", type=str, default="ckpts", help="Path to save trained model checkpoints")
    parser.add_argument("--vs_save_path", type=str, default="visualize", help="Path to save visualization outputs")
    args = parser.parse_args()
    main(args)