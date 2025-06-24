import argparse
import os
import random

import numpy as np
import torch
from my_models.degradation import LPDegradationModel
from PIL import Image
from my_models.cycle_gans import Generator
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from my_utils.utils import single2uint, uint2single


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    degradation = LPDegradationModel()
    model = Generator(in_channels=3, out_channels=3).to(device)

    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    transform = transforms.Compose(
        [
            transforms.Resize((args.height, args.width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    image_files = [
        f
        for f in os.listdir(args.input_folder)
        if os.path.isfile(os.path.join(args.input_folder, f))
    ]
    print(f"Found {len(image_files)} images to process...")

    with torch.no_grad():
        for filename in tqdm(image_files, desc="Processing images"):
            prob = random.random()
            if prob <= 0.4:
                try:
                    input_path = os.path.join(args.input_folder, filename)
                    img = Image.open(input_path).convert("RGB")
                    img_tensor = transform(img)
                    input_batch = img_tensor.unsqueeze(0).to(device)
                    output_tensor = model(input_batch)
                    output_path = os.path.join(args.output_folder, filename)
                    save_image(output_tensor, output_path, normalize=True)
                except Exception as e:
                    print(f"Could not process {filename}. Error: {e}")

            elif prob > 0.4 and prob <= 0.8:
                try:
                    input_path = os.path.join(args.input_folder, filename)
                    img = np.array(Image.open(input_path).convert("RGB"))
                    img = uint2single(img)
                    img = degradation.apply_degradation(img)
                    img = single2uint(img)
                    pil_img = Image.fromarray(img)
                    output_path = os.path.join(args.output_folder, filename)
                    pil_img.save(output_path)
                except Exception as e:
                    print(f"Could not process {filename}. Error: {e}")

            else:
                try:
                    input_path = os.path.join(args.input_folder, filename)
                    img = Image.open(input_path).convert("RGB")
                    img_tensor = transform(img)
                    input_batch = img_tensor.unsqueeze(0).to(device)
                    output_tensor = model(input_batch)
                    output_numpy = (
                        output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 0.5
                        + 0.5
                    )
                    img = degradation.apply_degradation(output_numpy)
                    img = single2uint(img)
                    pil_img = Image.fromarray(img)
                    output_path = os.path.join(args.output_folder, filename)
                    pil_img.save(output_path)
                except Exception as e:
                    print(f"Could not process {filename}. Error: {e}")

    print("Inference complete. Blurred images are saved in:", args.output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to the folder containing clear input images.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to the folder where blurred output images will be saved.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the saved generator checkpoint file.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=32,
        help="Image height the model was trained on.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=192,
        help="Image width the model was trained on.",
    )

    args = parser.parse_args()
    test(args)