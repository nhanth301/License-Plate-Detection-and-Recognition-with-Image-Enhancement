import torch
import torchvision.transforms as T
from PIL import Image
import argparse
import os 
import sys
sys.path.append(os.path.abspath('../models'))
from base_sp_lpr import LPSR  


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.ToTensor(),  # [0, 1]
    ])
    return transform(image).unsqueeze(0)  # shape: [1, C, H, W]

def save_image(tensor, output_path):
    tensor = tensor.squeeze(0).detach().cpu().clamp(0, 1)
    to_pil = T.ToPILImage()
    image = to_pil(tensor)
    image.save(output_path)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load image
    input_image = load_image(args.input).to(device)

    # Load model
    model = LPSR(
        num_channels=3,
        num_features=args.num_features,
        growth_rate=args.growth_rate,
        num_blocks=args.num_blocks,
        num_layers=args.num_layers,
        scale_factor=args.scale
    ).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print("No checkpoint loaded, using randomly initialized weights.")

    model.eval()
    with torch.no_grad():
        output = model(input_image)

    save_image(output, args.output)
    print(f"Saved output image to {args.output}")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="output.png", help="Path to save output image")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pth)")
    parser.add_argument("--scale", type=int, default=2, help="Upscaling factor (e.g., 2, 4)")
    parser.add_argument("--num_features", type=int, default=124)
    parser.add_argument("--growth_rate", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)

    args = parser.parse_args()
    return args

args = parse_opt()
if __name__ == "__main__":
    args = parse_opt()
    main(args)
