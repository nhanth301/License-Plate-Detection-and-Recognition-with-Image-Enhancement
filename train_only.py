import torch
import torch.optim as optim
import torch.nn as nn
from dataset.my_dataset import MyDataset
from torch.utils.data import DataLoader
import os 
import argparse
import matplotlib.pyplot as plt
from torch.nn import functional as F
from models.blur_generator import BlurGenerator
from models.discriminator import Discriminator
from torchvision import models
from tqdm import tqdm

def kernel_regularization_loss(kernel):
    tv_h = torch.sum(torch.abs(kernel[:, :, 1:, :] - kernel[:, :, :-1, :]))
    tv_w = torch.sum(torch.abs(kernel[:, :, :, 1:] - kernel[:, :, :, :-1]))
    return tv_h + tv_w

def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    alpha = torch.randn(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = torch.ones(d_interpolates.size(), device=device, requires_grad=False)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_gan(blur_generator, discriminator, dataloader, num_epochs, device, 
              save_path="models_gan_only", vs_save_path="visualize_gan_only", n_critic=5):

    torch.autograd.set_detect_anomaly(True)

    if not os.path.exists(vs_save_path):
        os.makedirs(vs_save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    l1_loss = nn.L1Loss()
    
    optimizer_blur_gen = optim.Adam(blur_generator.parameters(), lr=1e-4, betas=(0.0, 0.9))
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.0, 0.9))

    vgg = models.vgg16(pretrained=True).features.to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False
    vgg_content_layers = vgg[:22]

    best_gen_loss = float('inf')

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        blur_generator.train()
        discriminator.train()

        total_gen_loss = 0
        total_disc_loss = 0
        
        for batch_idx, (clear_img, blur_img) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)):
            clear_img, blur_img = clear_img.to(device), blur_img.to(device)
            
            optimizer_disc.zero_grad()
            
            fake_blur, _ = blur_generator(clear_img, blur_img)
            
            real_output = discriminator(blur_img)
            fake_output = discriminator(fake_blur.detach())

            lambda_gp = 10
            gradient_penalty = compute_gradient_penalty(discriminator, blur_img.data, fake_blur.data, device)
            disc_loss = torch.mean(fake_output) - torch.mean(real_output) + lambda_gp * gradient_penalty
            
            disc_loss.backward()
            optimizer_disc.step()
            
            total_disc_loss += disc_loss.item()

            if (batch_idx + 1) % n_critic == 0:
                optimizer_blur_gen.zero_grad()
                
                fake_blur, blur_kernel = blur_generator(clear_img, blur_img)
                fake_output_for_gen = discriminator(fake_blur)
                
                gen_adv_loss = -torch.mean(fake_output_for_gen)

                clear_feats_content = vgg_content_layers(clear_img)
                fake_feats_content = vgg_content_layers(fake_blur)
                content_loss =  l1_loss(fake_feats_content, clear_feats_content)
                
                k_reg_loss = kernel_regularization_loss(blur_kernel)
                
                total_gen_loss_step = gen_adv_loss + 0.05 * content_loss + 0.0001 * k_reg_loss
                
                total_gen_loss_step.backward()
                optimizer_blur_gen.step()
                
                total_gen_loss += total_gen_loss_step.item()

        avg_gen_loss = total_gen_loss / (len(dataloader) / n_critic) if total_gen_loss != 0 else 0
        avg_disc_loss = total_disc_loss / len(dataloader)

        print(f"Epoch {epoch+1}/{num_epochs}, Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}")

        if avg_gen_loss < best_gen_loss and avg_gen_loss != 0:
            best_gen_loss = avg_gen_loss
            torch.save(blur_generator.state_dict(), os.path.join(save_path, "best_blur_generator_gan.pth"))
            torch.save(discriminator.state_dict(), os.path.join(save_path, "best_discriminator_gan.pth"))
            print(f"Saved best models at epoch {epoch+1}")

        if (epoch + 1) % 20 == 0:
            torch.save(blur_generator.state_dict(), os.path.join(save_path, f"generator_gan_{epoch+1}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(save_path, f"discriminator_gan_{epoch+1}.pth"))

        with torch.no_grad():
            blur_generator.eval()
            
            fake_blur_vis, _ = blur_generator(clear_img, blur_img)

            clear_img_cpu = clear_img[0].detach().cpu()
            fake_blur_cpu = fake_blur_vis[0].detach().cpu()
            real_blur_cpu = blur_img[0].detach().cpu()

            def denormalize(img_tensor):
                img_tensor = img_tensor.clamp(0, 1)
                return img_tensor.permute(1, 2, 0).numpy()

            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1); plt.title("Clear Image (Input)"); plt.imshow(denormalize(clear_img_cpu)); plt.axis('off')
            plt.subplot(1, 3, 2); plt.title("Fake Blur"); plt.imshow(denormalize(fake_blur_cpu)); plt.axis('off')
            plt.subplot(1, 3, 3); plt.title("Real Blur"); plt.imshow(denormalize(real_blur_cpu)); plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"{vs_save_path}/output_epoch_{epoch+1}.png")
            plt.close()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MyDataset(args.clear_folder, args.blur_folder, image_size=(64, 128))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    blur_generator = BlurGenerator(
        in_channels=3, 
        feature_dim=64, 
        kernel_size=args.kernel_size
    ).to(device)
    
    discriminator = Discriminator(in_channels=3).to(device)

    train_gan(
        blur_generator, discriminator, dataloader, 
        args.epochs, device, 
        save_path=args.save_path, 
        vs_save_path=args.vs_save_path,
        n_critic=args.n_critic
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GAN for Realistic Image Blurring")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--kernel_size", type=int, default=11, help="Size of the blur kernel to estimate")
    parser.add_argument("--n_critic", type=int, default=5, help="Number of critic updates per generator update")
    parser.add_argument("--clear_folder", type=str, required=True, help="Path to clear image folder")
    parser.add_argument("--blur_folder", type=str, required=True, help="Path to blurred image folder")
    parser.add_argument("--save_path", type=str, default="ckpts_gan_only", help="Path to save trained model checkpoints")
    parser.add_argument("--vs_save_path", type=str, default="visualize_gan_only", help="Path to save visualization outputs")
    args = parser.parse_args()
    main(args)