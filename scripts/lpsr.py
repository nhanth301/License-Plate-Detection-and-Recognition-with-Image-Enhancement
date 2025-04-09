import os
import sys
import math
import argparse
import torch
import torchvision.transforms as T
from PIL import Image
from loguru import logger
from tqdm import tqdm

# Add model path
sys.path.append(os.path.abspath('../models'))
from base_sp_lpr import LPSR


def load_image(image_path):
    """Load and preprocess an image from the given path.

    Args:
        image_path (str): Path to the input image.

    Returns:
        torch.Tensor or None: Transformed image tensor of shape [C, H, W], or None if loading fails.

    Example:
        >>> img_tensor = load_image("path/to/image.png")
    """
    try:
        image = Image.open(image_path).convert("RGB").resize((96, 32), Image.BICUBIC)
        transform = T.Compose([T.ToTensor()])
        return transform(image)
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        return None


def save_image(tensor, output_path):
    """Save a tensor image to disk.

    Args:
        tensor (torch.Tensor): Image tensor with values in [0, 1].
        output_path (str): Path to save the output image.
    """
    tensor = tensor.detach().cpu().clamp(0, 1)
    to_pil = T.ToPILImage()
    image = to_pil(tensor)
    image.save(output_path)


def collect_images(folder):
    """Collect and load valid image files from a directory.

    Args:
        folder (str): Path to the input directory.

    Returns:
        tuple[list[torch.Tensor], list[str]]: Tuple of loaded image tensors and their file paths.

    Example:
        >>> imgs, paths = collect_images("input_folder/")
    """
    valid_images = []
    valid_paths = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in image_extensions):
            img = load_image(file_path)
            if img is not None:
                valid_images.append(img)
                valid_paths.append(file_path)

    return valid_images, valid_paths


def process_batch(model, batch, device):
    """Apply the model to a batch of images without gradient tracking.

    Args:
        model (torch.nn.Module): Super-resolution model.
        batch (torch.Tensor): Batch of input images.
        device (torch.device): Target device.

    Returns:
        torch.Tensor: Model output.
    """
    with torch.no_grad():
        batch_tensor = batch.to(device)
        return model(batch_tensor)


def main(args):
    """Main processing pipeline for loading model and applying it to images in batches.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.input_folder) or not os.path.isdir(args.input_folder):
        logger.error(f"Input folder {args.input_folder} does not exist or is not a directory")
        return

    os.makedirs(args.output_folder, exist_ok=True)

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
        logger.info(f"Loaded checkpoint from {args.checkpoint}")
    else:
        logger.warning("No checkpoint loaded, using randomly initialized weights.")

    model.eval()

    # Load images
    logger.info(f"Collecting images from {args.input_folder}...")
    images, image_paths = collect_images(args.input_folder)

    if not images:
        logger.error("No valid images found in the input folder")
        return

    logger.info(f"Found {len(images)} valid images")

    batch_size = args.batch_size
    num_batches = math.ceil(len(images) / batch_size)
    processed_count = 0

    logger.info(f"Processing images in batches of {batch_size}...")

    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(images))

        current_batch = torch.stack(images[start_idx:end_idx])
        current_paths = image_paths[start_idx:end_idx]

        outputs = process_batch(model, current_batch, device)

        for j, output in enumerate(outputs):
            original_filename = os.path.basename(current_paths[j])
            output_path = os.path.join(args.output_folder, original_filename)
            save_image(output, output_path)
            processed_count += 1

    logger.info(f"Processing complete. Successfully processed {processed_count}/{len(images)} images.")
    logger.info(f"Output saved to {args.output_folder}")


def parse_opt():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True, help="Path to input folder containing images")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to output folder for saving processed images")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pth)")
    parser.add_argument("--scale", type=int, default=2, help="Upscaling factor (e.g., 2, 4)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")
    parser.add_argument("--num_features", type=int, default=124, help="Number of features in the model")
    parser.add_argument("--growth_rate", type=int, default=64, help="Growth rate of dense blocks")
    parser.add_argument("--num_blocks", type=int, default=8, help="Number of residual dense blocks")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers per block")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_opt()
    main(args)
