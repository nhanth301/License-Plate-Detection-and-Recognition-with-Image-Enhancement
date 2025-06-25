# file: evaluation.py

import argparse
import os
import sys
from typing import List, Tuple

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image
from tqdm import tqdm
import Levenshtein
from torchvision import transforms
from my_models.lpsr import LPSR
from my_models.detection import Detection
from my_utils.utils import sort_license_plate_detections




def get_ground_truth_from_filename(filename: str) -> str:
    """Extracts the ground truth license plate text from the image filename."""
    return os.path.splitext(filename)[0].upper()


def calculate_cer(ground_truth: str, ocr_result: str) -> float:
    """Calculates the Character Error Rate (CER) using Levenshtein distance."""
    if not ground_truth:
        return 1.0 if ocr_result else 0.0
    distance = Levenshtein.distance(ground_truth, ocr_result)
    return distance / len(ground_truth)


def preprocess_for_sr(
    plate_image: np.ndarray, target_size: Tuple[int, int] = (192, 32)
) -> torch.Tensor:
    """Preprocesses a plate image for the Super-Resolution model."""
    rgb_img = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img).resize(target_size, Image.BICUBIC)
    transform = transforms.Compose([
        transforms.Resize(size=(target_size[1], target_size[0])),
        transforms.ToTensor(),
    ])
    return transform(pil_img).unsqueeze(0)


def run_ocr(model: Detection, image: np.ndarray) -> str:
    """Runs the OCR model on a given image and returns the sorted text."""
    char_results, _ = model.detect(image, bb_scale=False)
    sorted_chars = sort_license_plate_detections(char_results)
    return "".join([char_name.upper() for char_name, _, _ in sorted_chars])



def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate Super-Resolution model performance for OCR.")
    parser.add_argument("--eval-folder", type=str, required=True, help="Path to the folder with evaluation images.")
    parser.add_argument("--sr-weights", type=str, required=True, help="Path to the Super-Resolution model weights (.pth).")
    parser.add_argument("--ocr-weights", type=str, required=True, help="Path to the OCR model weights (.pt).")
    parser.add_argument("--imgsz-ocr", nargs="+", type=int, default=[128, 128], help="OCR model inference size (h, w).")
    parser.add_argument("--ocr-conf", type=float, default=0.45, help="OCR confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.3, help="NMS IoU threshold for OCR.")
    parser.add_argument("--device", default="cuda", help="Computation device: 'cuda' or 'cpu'.")
    return parser.parse_args()


def main(args: argparse.Namespace):
    """Main function to run the evaluation pipeline."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load Models ---
    logger.info("Loading Super-Resolution model...")
    sr_model = LPSR(num_channels=3, num_features=32, growth_rate=16, num_blocks=4, num_layers=4, scale_factor=None).to(device)
    checkpoint = torch.load(args.sr_weights, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    sr_model.load_state_dict(state_dict)
    sr_model.eval()
    logger.info("Super-Resolution model loaded.")

    logger.info("Loading OCR model...")
    ocr_model = Detection(size=args.imgsz_ocr, weights_path=args.ocr_weights, device=device, iou_thres=args.iou, conf_thres=args.ocr_conf)
    logger.info("OCR model loaded.")
    
    # --- Data Preparation ---
    image_files = [f for f in os.listdir(args.eval_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        logger.error(f"No images found in evaluation folder: {args.eval_folder}")
        return

    logger.info(f"Found {len(image_files)} images for evaluation.")

    # --- Evaluation Loop ---
    results = {
        "without_sr": {"total_cer": 0, "exact_matches": 0},
        "with_sr": {"total_cer": 0, "exact_matches": 0}
    }
    
    for filename in tqdm(image_files, desc="Evaluating Images"):
        image_path = os.path.join(args.eval_folder, filename)
        ground_truth = get_ground_truth_from_filename(filename)
        
        original_image = cv2.imread(image_path)
        if original_image is None:
            logger.warning(f"Could not read image: {filename}. Skipping.")
            continue

        # --- Path 1: OCR on Original Low-Resolution Image ---
        original_ocr_text = run_ocr(ocr_model, original_image.copy())
        
        # --- Path 2: OCR on Super-Resolved Image ---
        with torch.no_grad():
            sr_input_tensor = preprocess_for_sr(original_image).to(device)
            sr_output_tensor = sr_model(sr_input_tensor).squeeze(0).cpu()
            sr_np = sr_output_tensor.permute(1, 2, 0).numpy()
            sr_np = np.clip(sr_np, 0, 1) # Ensure values are in [0, 1] range
            sr_image_bgr = cv2.cvtColor((sr_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        sr_ocr_text = run_ocr(ocr_model, sr_image_bgr.copy())

        # --- Calculate and Store Metrics ---
        results["without_sr"]["total_cer"] += calculate_cer(ground_truth, original_ocr_text)
        if ground_truth == original_ocr_text:
            results["without_sr"]["exact_matches"] += 1
            
        results["with_sr"]["total_cer"] += calculate_cer(ground_truth, sr_ocr_text)
        if ground_truth == sr_ocr_text:
            results["with_sr"]["exact_matches"] += 1

        logger.info(f"File: {filename} | GT: {ground_truth} | Original OCR: {original_ocr_text} | SR OCR: {sr_ocr_text}")

    # --- Print Final Report ---
    num_images = len(image_files)
    logger.info("\n--- SR Model Evaluation Report ---")
    logger.info(f"Total images evaluated: {num_images}")
    
    # Without SR Results
    wosr_accuracy = (results["without_sr"]["exact_matches"] / num_images) * 100
    wosr_cer = (results["without_sr"]["total_cer"] / num_images) * 100
    logger.info("\n--- Without Super-Resolution ---")
    logger.info(f"Correct Full Plates: {results['without_sr']['exact_matches']}/{num_images} ({wosr_accuracy:.2f}%)")
    logger.info(f"Average Character Error Rate (CER): {wosr_cer:.2f}%")

    # With SR Results
    wsr_accuracy = (results["with_sr"]["exact_matches"] / num_images) * 100
    wsr_cer = (results["with_sr"]["total_cer"] / num_images) * 100
    logger.info("\n--- With Super-Resolution ---")
    logger.info(f"Correct Full Plates: {results['with_sr']['exact_matches']}/{num_images} ({wsr_accuracy:.2f}%)")
    logger.info(f"Average Character Error Rate (CER): {wsr_cer:.2f}%")
    
    # Summary of Improvement
    accuracy_improvement = wsr_accuracy - wosr_accuracy
    cer_reduction = wosr_cer - wsr_cer
    logger.info("\n--- Summary of Improvement ---")
    logger.info(f"Full Plate Accuracy Improvement: {accuracy_improvement:+.2f}%")
    logger.info(f"Character Error Rate Reduction: {cer_reduction:+.2f}%")
    logger.info("---------------------------------")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)