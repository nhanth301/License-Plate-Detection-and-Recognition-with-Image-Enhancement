import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from my_models.lpsr import LPSR
from dataset.lpsr_dataset import ImageDataset


def get_fixed_samples(dataloader, num_samples=3):
    lrs, hrs = [], []
    for lr_batch, hr_batch in dataloader:
        for i in range(lr_batch.size(0)):
            if len(lrs) < num_samples:
                lrs.append(lr_batch[i])
                hrs.append(hr_batch[i])
            else:
                break
        if len(lrs) >= num_samples:
            break
    return torch.stack(lrs), torch.stack(hrs)


def prepare_test_images(image_paths, transform):
    tensors = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            tensors.append(transform(img))
        except FileNotFoundError:
            print(f"Warning: Test image not found at {path}")
            continue
    return torch.stack(tensors) if tensors else None


def denormalize(tensor):
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.cpu().numpy().transpose(1, 2, 0)


def visualize_results(model, fixed_samples, device, epoch, save_dir):
    model.eval()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num_rows = sum(
        len(samples[0]) for samples in fixed_samples.values() if samples[0] is not None
    )
    if num_rows == 0:
        return

    fig, axs = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
    if num_rows == 1:
        axs = axs.reshape(1, -1)
    fig.suptitle(f"Epoch {epoch+1}", fontsize=16)

    row_idx = 0
    with torch.no_grad():
        for name, (lr_tensors, hr_tensors) in fixed_samples.items():
            if lr_tensors is None:
                continue

            lr_tensors = lr_tensors.to(device)
            sr_tensors = model(lr_tensors)

            for i in range(lr_tensors.size(0)):
                lr_img = denormalize(lr_tensors[i])
                sr_img = denormalize(sr_tensors[i])
                hr_img = denormalize(hr_tensors[i]) if hr_tensors is not None else None

                axs[row_idx, 0].imshow(lr_img)
                axs[row_idx, 0].set_title(f"{name} Sample - Original LR")
                axs[row_idx, 0].axis("off")

                axs[row_idx, 1].imshow(sr_img)
                axs[row_idx, 1].set_title("Super-Resolved")
                axs[row_idx, 1].axis("off")

                if hr_img is not None:
                    axs[row_idx, 2].imshow(hr_img)
                    axs[row_idx, 2].set_title("Ground Truth HR")
                axs[row_idx, 2].axis("off")
                row_idx += 1

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch+1}_visualization.png"))
    plt.close()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose(
        [
            transforms.Resize((args.height, args.width)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = ImageDataset(
        hr_dir=args.hr_train_dir, lr_dir=args.lr_train_dir, transform=transform
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    val_dataset = ImageDataset(
        hr_dir=args.hr_val_dir, lr_dir=args.lr_val_dir, transform=transform
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = LPSR(
        num_channels=3,
        num_features=32,
        growth_rate=16,
        num_blocks=4,
        num_layers=4,
        scale_factor=1,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )

    fixed_train_lr, fixed_train_hr = get_fixed_samples(train_dataloader, num_samples=2)
    fixed_val_lr, fixed_val_hr = get_fixed_samples(val_dataloader, num_samples=2)
    fixed_test_lr = prepare_test_images(args.test_image_paths, transform)
    fixed_test_hr = (
        torch.zeros_like(fixed_test_lr) if fixed_test_lr is not None else None
    )

    fixed_samples = {
        "Train": (fixed_train_lr, fixed_train_hr),
        "Validation": (fixed_val_lr, fixed_val_hr),
        "Test": (fixed_test_lr, fixed_test_hr),
    }

    best_psnr = -float("inf")
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for epoch in range(args.epochs):
        model.train()
        train_bar = tqdm(
            train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]"
        )

        for lr_images, hr_images in train_bar:
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)
            optimizer.zero_grad()
            sr_images = model(lr_images)
            loss = criterion(sr_images, hr_images)
            loss.backward()
            optimizer.step()
            train_bar.set_postfix(loss=loss.item())

        model.eval()
        total_psnr, count = 0, 0
        with torch.no_grad():
            for lr_images, hr_images in val_dataloader:
                lr_images, hr_images = lr_images.to(device), hr_images.to(device)
                sr_images = model(lr_images)
                sr_images = torch.clamp(sr_images, 0, 1)

                sr_images_np = sr_images.cpu().numpy()
                hr_images_np = hr_images.cpu().numpy()

                for i in range(sr_images_np.shape[0]):
                    psnr = peak_signal_noise_ratio(
                        hr_images_np[i], sr_images_np[i], data_range=1.0
                    )
                    total_psnr += psnr
                    count += 1

        avg_psnr = total_psnr / count
        print(f"Epoch [{epoch + 1}/{args.epochs}] - Validation Avg PSNR: {avg_psnr:.4f}")

        scheduler.step(avg_psnr)

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print(f"Model saved with new best PSNR: {best_psnr:.4f}")

        visualize_results(model, fixed_samples, device, epoch, save_dir=output_dir)

    print("Training Complete!")
    torch.save(model.state_dict(), os.path.join(output_dir, "last_model.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hr_train_dir", type=str, required=True)
    parser.add_argument("--lr_train_dir", type=str, required=True)
    parser.add_argument("--hr_val_dir", type=str, required=True)
    parser.add_argument("--lr_val_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--test_image_paths", type=str, nargs="+", default=[])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=32)
    args = parser.parse_args()
    main(args)