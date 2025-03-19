import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse

from PIL import Image
from models.degradation import LPDegradationModel
import numpy as np
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_lr(input_path, output_path):
    """
    Creates low-resolution (LR) and high-resolution (HR) image pairs.

    Args:
        input_path (str): Path to the input image or directory.
        output_path (str): Path to the output directory.
    """
    hr_output_dir = os.path.join(output_path, 'HR')
    lr_output_dir = os.path.join(output_path, 'LR')
    os.makedirs(hr_output_dir, exist_ok=True)
    os.makedirs(lr_output_dir, exist_ok=True)

    degradation = LPDegradationModel()

    if os.path.isdir(input_path):
        filenames = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        logging.info(f"Processing {len(filenames)} images from directory: {input_path}")
        for filename in tqdm(filenames, desc="Processing images"):
            try:
                img_path = os.path.join(input_path, filename)
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception as e:
                    logging.error(f"Error opening image {filename}: {e}")
                    continue

                lr_img_array = degradation.apply_degradation(np.array(img))
                lr_img = Image.fromarray(lr_img_array.astype(np.uint8))

                lr_img.save(os.path.join(lr_output_dir, filename))
                img.save(os.path.join(hr_output_dir, filename))
            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")
    else:
        try:
            img = Image.open(input_path).convert("RGB")
        except Exception as e:
            logging.error(f"Error opening image {input_path}: {e}")
            return

        lr_img_array = degradation.apply_degradation(np.array(img))
        lr_img = Image.fromarray(lr_img_array.astype(np.uint8))

        lr_img.save(os.path.join(lr_output_dir, os.path.basename(input_path)))
        img.save(os.path.join(hr_output_dir, os.path.basename(input_path)))
        logging.info(f"Processed single image: {input_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create LR/HR image pairs.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image or directory")
    parser.add_argument("--output", type=str, required=True, help="Path to the output directory")
    args = parser.parse_args()
    logging.info(f"Starting create_lr with arguments: {args}")
    create_lr(args.input, args.output)
    logging.info("Finished create_lr")
