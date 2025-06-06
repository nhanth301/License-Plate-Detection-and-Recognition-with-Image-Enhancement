import torch
import torch.optim as optim
import torch.nn as nn
from dataset.my_dataset import MyDataset
from torch.utils.data import DataLoader
import os 
import argparse


from models.feature_extractor import FeatureExtractor
from models.clear_generator import LPSR
from models.blur_generator import BlurGenerator
from models.discriminator import Discriminator



def train_models(extractor, blur_generator, discriminator, clear_generator, dataloader, num_epochs, device, save_path="models"):
    os.makedirs(save_path, exist_ok=True)

    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    optimizer_blur_gen = optim.adam.Adam(blur_generator.parameters(), lr=0.0002)
    optimizer_clear_gen = optim.adam.Adam(clear_generator.parameters(), lr=0.0002)
    optimizer_disc = optim.adam.Adam(discriminator.parameters(), lr=0.0002)
    
    best_loss = float('inf')
    for epoch in range(num_epochs):
        extractor.train()
        blur_generator.train()
        clear_generator.train()
        discriminator.train()
        total_gen_loss = 0
        total_disc_loss = 0
        total_sr_loss = 0
        
        for clear_img, blur_img in dataloader:
            clear_img, blur_img = clear_img.to(device), blur_img.to(device)
            batch_size = clear_img.size(0)
            
            # Train Feature Extractor
            blur_features = extractor(clear_img, blur_img)

            
            # Train Discriminator
            fake_blur = blur_generator(clear_img, blur_features)
            real_labels = torch.ones(batch_size).to(device)
            fake_labels = torch.zeros(batch_size).to(device)
            
            # Discriminator on real blur
            real_output = discriminator(blur_img)
            disc_real_loss = bce_loss(real_output, real_labels)
            
            # Discriminator on fake blur
            fake_output = discriminator(fake_blur)
            disc_fake_loss = bce_loss(fake_output, fake_labels)
            
            disc_loss = (disc_real_loss + disc_fake_loss) / 2
            optimizer_disc.zero_grad()
            disc_loss.backward()
            optimizer_disc.step()
            total_disc_loss += disc_loss.item()

            
            # Train Generator
            fake_output = discriminator(fake_blur)
            gen_adv_loss = bce_loss(fake_output, real_labels)  # Fool discriminator
            gen_loss = gen_adv_loss
            
            optimizer_blur_gen.zero_grad()
            gen_loss.backward()
            optimizer_blur_gen.step()
            total_gen_loss += gen_loss.item()


            hr_output = clear_generator(fake_blur)
            sr_loss = l1_loss(hr_output, clear_img) + l2_loss(hr_output, clear_img)
            optimizer_clear_gen.zero_grad()
            sr_loss.backward()
            optimizer_clear_gen.step()
            total_sr_loss += sr_loss.item()
        
        avg_gen_loss = total_gen_loss / len(dataloader)
        avg_disc_loss = total_disc_loss / len(dataloader)
        avg_sr_loss = total_sr_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}, SR Loss: {avg_sr_loss:.4f}")
        
        # Save best models
        if avg_gen_loss < best_loss:
            best_loss = avg_gen_loss
            torch.save(extractor.state_dict(), os.path.join(save_path, "extractor.pth"))
            torch.save(blur_generator.state_dict(), os.path.join(save_path, "generator.pth"))
            torch.save(discriminator.state_dict(), os.path.join(save_path, "discriminator.pth"))
            torch.save(clear_generator.state_dict(), os.path.join(save_path, "sr.pth"))
            print(f"Saved best models at epoch {epoch+1}")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset and dataloader
    dataset = MyDataset(args.clear_folder, args.blur_folder, image_size=(64, 128))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize models
    extractor = FeatureExtractor(in_channels=3, feature_dim=64).to(device)
    blur_generator = BlurGenerator(in_channels=3, out_channels=3, feature_dim=64).to(device)
    clear_generator = LPSR(num_channels=3,
                           num_features=32,
                           growth_rate=16,
                           num_blocks=4,
                           num_layers=4,
                           scale_factor=None).to(device)
    discriminator = Discriminator().to(device)

    # Train models
    train_models(extractor, blur_generator, discriminator, clear_generator,
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