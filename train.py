import torch
import torch.optim as optim
import torch.nn as nn
from dataset.my_dataset import MyDataset
from torch.utils.data import DataLoader
import os 
import argparse
import matplotlib.pyplot as plt

from models.feature_extractor import FeatureExtractor
from models.clear_generator import LPSR
from models.blur_generator import BlurGenerator
from models.discriminator import Discriminator



def train_models(blur_generator, discriminator, clear_generator,
                 dataloader, num_epochs, device, save_path="models"):

    os.makedirs(save_path, exist_ok=True)
    if not os.path.exists("train_outputs"):
            os.makedirs("train_outputs")
    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()

    optimizer_blur_gen = optim.Adam(blur_generator.parameters(), lr=0.0002)
    optimizer_clear_gen = optim.Adam(clear_generator.parameters(), lr=0.0002)
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=0.0002)
    
    best_loss = float('inf')

    for epoch in range(num_epochs):
        blur_generator.train()
        clear_generator.train()
        discriminator.train()

        total_gen_loss = 0
        total_disc_loss = 0
        total_sr_loss = 0

        for batch_idx, (clear_img, blur_img) in enumerate(dataloader):
            clear_img, blur_img = clear_img.to(device), blur_img.to(device)
            batch_size = clear_img.size(0)

            if batch_idx % 2 == 0:
                fake_blur = blur_generator(clear_img, blur_img)

                # Label smoothing cho real labels
                real_labels = torch.full((batch_size, 1), 0.9, device=device)
                fake_labels = torch.zeros((batch_size, 1), device=device)

                # Thêm noise Gaussian nhỏ vào đầu vào discriminator
                real_input = blur_img + 0.05 * torch.randn_like(blur_img)
                fake_input = fake_blur.detach() + 0.05 * torch.randn_like(fake_blur)

                # Discriminator trên ảnh blur thật
                real_output = discriminator(real_input)
                disc_real_loss = bce_loss(real_output, real_labels)

                # Discriminator trên ảnh blur giả
                fake_output = discriminator(fake_input)
                disc_fake_loss = bce_loss(fake_output, fake_labels)

                disc_loss = (disc_real_loss + disc_fake_loss) / 2

                optimizer_disc.zero_grad()
                disc_loss.backward()
                # Gradient clipping cho discriminator
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                optimizer_disc.step()
                total_disc_loss += disc_loss.item()
            else:
                disc_loss = torch.tensor(0.0).to(device)  # không cập nhật discriminator batch này

            # ======= Train Blur Generator =======
            fake_blur = blur_generator(clear_img, blur_img)
            gen_labels = torch.full((batch_size, 1), 0.9, device=device)  # label smoothing cho generator fool discriminator
            fake_output_for_gen = discriminator(fake_blur.detach())
            gen_adv_loss = bce_loss(fake_output_for_gen, gen_labels)
            gen_loss = gen_adv_loss

            optimizer_blur_gen.zero_grad()
            gen_loss.backward()
            optimizer_blur_gen.step()
            total_gen_loss += gen_loss.item()

            # ======= Train Clear Generator (SR) =======
            hr_output = clear_generator(fake_blur)
            sr_loss = l1_loss(hr_output, clear_img) + l2_loss(hr_output, clear_img)
            optimizer_clear_gen.zero_grad()
            sr_loss.backward()
            optimizer_clear_gen.step()
            total_sr_loss += sr_loss.item()

        avg_gen_loss = total_gen_loss / len(dataloader)
        avg_disc_loss = total_disc_loss / (len(dataloader)//2)  # discriminator chỉ update mỗi 2 batch
        avg_sr_loss = total_sr_loss / len(dataloader)

        print(f"Epoch {epoch+1}/{num_epochs}, Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}, SR Loss: {avg_sr_loss:.4f}")

        # Visualize hình ảnh (lấy batch đầu tiên của epoch)
        clear_img_cpu = clear_img[0].detach().cpu()
        fake_blur_cpu = fake_blur[0].detach().cpu()
        real_blur_cpu = blur_img[0].detach().cpu()
        sr_img_cpu = hr_output[0].detach().cpu()

        # Chuyển về [H,W,C] và range 0-1 cho matplotlib nếu cần (giả sử ảnh đã normalize [-1,1])
        def denormalize(img_tensor):
            img_tensor = img_tensor.clamp(0, 1)
            return img_tensor.permute(1, 2, 0).numpy()

        plt.figure(figsize=(16,4))  # rộng hơn để đủ chỗ cho 4 ảnh

        plt.subplot(1,4,1)
        plt.title("Clear Image (GT)")
        plt.imshow(denormalize(clear_img_cpu))
        plt.axis('off')

        plt.subplot(1,4,2)
        plt.title("Fake Blur")
        plt.imshow(denormalize(fake_blur_cpu))
        plt.axis('off')

        plt.subplot(1,4,3)
        plt.title("SR Image")
        plt.imshow(denormalize(sr_img_cpu))
        plt.axis('off')

        plt.subplot(1,4,4)
        plt.title("Real Blur")
        plt.imshow(denormalize(real_blur_cpu))  # bạn cần có biến real_blur_cpu chuẩn bị sẵn
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(f"train_outputs/output_epoch_{epoch}.png")
        plt.close()  # đóng figure để tránh tốn bộ nhớ

        # Save best models (dựa trên gen loss)
        if avg_gen_loss < best_loss:
            best_loss = avg_gen_loss
            torch.save(blur_generator.state_dict(), os.path.join(save_path, "generator.pth"))
            torch.save(discriminator.state_dict(), os.path.join(save_path, "discriminator.pth"))
            torch.save(clear_generator.state_dict(), os.path.join(save_path, "sr.pth"))
            print(f"Saved best models at epoch {epoch+1}")

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
                 dataloader, args.epochs, device, save_path=args.save_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image deblurring model")

    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--clear_folder", type=str, help="Path to clear image folder")
    parser.add_argument("--blur_folder", type=str, help="Path to blurred image folder")
    parser.add_argument("--save_path", type=str, default="ckpts", help="Path to save trained model checkpoints")

    args = parser.parse_args()
    main(args)