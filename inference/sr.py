import argparse
import os
import sys
from typing import Tuple

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from my_models.lpsr import LPSR
from my_models.cycle_gans import Generator


def preprocess_image(
    image_path: str, target_size: Tuple[int, int]
) -> torch.Tensor:
    """
    Loads an image, resizes it, and converts it to a PyTorch tensor.

    Args:
        image_path (str): The path to the input image.
        target_size (Tuple[int, int]): The target size as (Width, Height).

    Returns:
        torch.Tensor: The preprocessed image tensor ready for the model.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((target_size[1], target_size[0])),  # H, W
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        # Add a batch dimension (B, C, H, W)
        return transform(image).unsqueeze(0)
    except FileNotFoundError:
        logger.error(f"Input image not found at: {image_path}")
        return None
    except Exception as e:
        logger.error(f"Could not process image {image_path}: {e}")
        return None


def postprocess_output(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts the output tensor from the SR model back to a saveable image format.

    Args:
        tensor (torch.Tensor): The output tensor from the model.

    Returns:
        np.ndarray: The final image in BGR format for saving with OpenCV.
    """
    # Remove batch dimension, move to CPU, and convert to NumPy
    sr_numpy = tensor.squeeze(0).cpu().numpy() * 0.5 + 0.5  # Undo normalization
    
    # Clip values to ensure they are in the valid [0, 1] range
    sr_numpy = np.clip(sr_numpy, 0, 1)
    
    # Transpose from (C, H, W) to (H, W, C)
    sr_numpy_hwc = np.transpose(sr_numpy, (1, 2, 0))
    
    # Scale from [0, 1] to [0, 255] and convert to integer type
    sr_image_uint8 = (sr_numpy_hwc * 255).astype(np.uint8)
    
    # Convert from RGB (standard for PIL/PyTorch) to BGR (standard for OpenCV)
    sr_image_bgr = cv2.cvtColor(sr_image_uint8, cv2.COLOR_RGB2BGR)
    
    return sr_image_bgr


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the inference script."""
    parser = argparse.ArgumentParser(description="Run Super-Resolution inference on a folder of images.")
    parser.add_argument("--weights", type=str, required=True, help="Path to the Super-Resolution model weights (.pth).")
    parser.add_argument("--input-folder", type=str, required=True, help="Path to the folder containing low-resolution images.")
    parser.add_argument("--output-folder", type=str, required=True, help="Path to the folder where super-resolved images will be saved.")
    parser.add_argument("--height", type=int, default=32, help="Image height expected by the model.")
    parser.add_argument("--width", type=int, default=192, help="Image width expected by the model.")
    parser.add_argument("--device", default="cuda", help="Computation device: 'cuda' or 'cpu'.")
    return parser.parse_args()


def main(args: argparse.Namespace):
    """Main function to run the batch inference process."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load Model ---
    logger.info(f"Loading Super-Resolution model from {args.weights}...")
    try:
        # model = LPSR(num_channels=3, num_features=32, growth_rate=16, num_blocks=4, num_layers=4, scale_factor=None).to(device)
        model = Generator().to(device)  # Assuming Generator is the correct model class
        checkpoint = torch.load(args.weights, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        model.eval()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # --- Prepare Directories ---
    os.makedirs(args.output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(args.input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if not image_files:
        logger.warning(f"No images found in the input folder: {args.input_folder}")
        return

    logger.info(f"Found {len(image_files)} images to process. Results will be saved to {args.output_folder}")

    # --- Inference Loop ---
    with torch.no_grad():
        for filename in tqdm(image_files, desc="Processing Images"):
            input_path = os.path.join(args.input_folder, filename)
            output_path = os.path.join(args.output_folder, filename)

            # Preprocess the image
            input_tensor = preprocess_image(input_path, target_size=(args.width, args.height))
            if input_tensor is None:
                continue
            
            input_tensor = input_tensor.to(device)

            # Run inference
            output_tensor = model(input_tensor)

            # Postprocess and save the output
            sr_image = postprocess_output(output_tensor)
            cv2.imwrite(output_path, sr_image)

    logger.info("Inference complete.")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)