# file: find_improvements.py

import argparse
import os
import sys
import random
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from my_models.lpsr import LPSR
from my_models.detection import Detection
from my_utils.utils import sort_license_plate_detections


# --- Helper Functions ---

def get_ground_truth_from_filename(filename: str) -> str:
    """Extracts the ground truth license plate text from the image filename."""
    return os.path.splitext(filename)[0].upper()


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



def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Applies specialized preprocessing steps for OCR on a license plate image.
    These steps aim to improve character clarity and contrast.

    Args:
        image (np.ndarray): The cropped license plate image in BGR format.

    Returns:
        np.ndarray: The preprocessed image, still in BGR format to be
                    compatible with the detection model's input.
    """
    # 1. Convert to grayscale, as color is not needed for OCR
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    # This is very effective for images with uneven lighting or low contrast.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced_gray = clahe.apply(gray)

    # 3. Convert the single-channel enhanced grayscale image back to a 3-channel BGR image.
    # This is necessary because the YOLO-based Detection model expects a 3-channel input.
    preprocessed_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    return preprocessed_bgr


def run_ocr(model: Detection, image: np.ndarray) -> Tuple[str, float]:
    """
    Runs the OCR model on a given image and returns the sorted text
    and the average confidence score.
    """
    preprocessed_image = preprocess_for_ocr(image)
    char_results, _ = model.detect(preprocessed_image, bb_scale=False)
    if not char_results:
        return "", 0.0

    sorted_chars_with_conf = sort_license_plate_detections(char_results)
    if not sorted_chars_with_conf:
        return "", 0.0

    ocr_text = "".join([char_info[0].upper() for char_info in sorted_chars_with_conf])
    confidences = [float(char_info[1]) for char_info in sorted_chars_with_conf]
    avg_confidence = sum(confidences) / len(confidences)
    
    return ocr_text, avg_confidence

def create_visualization_panel(results: List[Dict[str, Any]], output_path: str):
    """
    Creates a professional report-style image panel using OpenCV to visualize
    the "Before vs. After" effect of the Super-Resolution model.
    """
    if not results:
        logger.warning("No improvement cases found to visualize.")
        return

    # --- Layout Constants ---
    NUM_COLS = 3
    CASE_W, CASE_H = 420, 320  # Adjusted for vertical layout
    PADDING = 20
    HEADER_H = 60
    COLORS = {
        "bg": (45, 45, 45), "case_bg": (60, 60, 60), "header": (220, 220, 220),
        "success": (110, 255, 110), "fail": (100, 100, 255), "gt": (255, 200, 120)
    }
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    num_cases = len(results)
    num_rows = (num_cases + NUM_COLS - 1) // NUM_COLS

    canvas_w = (CASE_W * NUM_COLS) + (PADDING * (NUM_COLS + 1))
    canvas_h = HEADER_H + (CASE_H * num_rows) + (PADDING * (num_rows))
    canvas = np.full((canvas_h, canvas_w, 3), COLORS["bg"], dtype=np.uint8)
    
    # --- Draw Main Header ---
    title_text = "Super-Resolution OCR Improvement Analysis"
    (text_w, text_h), _ = cv2.getTextSize(title_text, FONT, 1.2, 2)
    title_x = (canvas_w - text_w) // 2
    cv2.putText(canvas, title_text, (title_x, HEADER_H - 20),
                FONT, 1.2, COLORS["header"], 2)

    # --- Helper to resize image to a fixed width, maintaining aspect ratio ---
    def resize_to_width(img, width):
        h, w = img.shape[:2]
        ratio = width / w
        return cv2.resize(img, (width, int(h * ratio)), interpolation=cv2.INTER_AREA)

    # --- Draw each case ---
    for i, case in enumerate(results):
        row, col = i // NUM_COLS, i % NUM_COLS
        x_start = PADDING + col * (CASE_W + PADDING)
        y_start = HEADER_H + PADDING + row * (CASE_H + PADDING)

        # Draw case background and border
        cv2.rectangle(canvas, (x_start, y_start), (x_start + CASE_W, y_start + CASE_H), COLORS["case_bg"], -1)
        cv2.rectangle(canvas, (x_start, y_start), (x_start + CASE_W, y_start + CASE_H), (100, 100, 100), 1)

        # --- Draw Content Inside Each Case Box ---
        current_y = y_start + 35
        content_x = x_start + 15
        
        # Ground Truth Text
        cv2.putText(canvas, f"Ground Truth: {case['ground_truth']}", (content_x, current_y),
                    FONT, 0.8, COLORS["gt"], 2)
        current_y += 40

        # Original Image and its OCR Result
        img_before = resize_to_width(case["original_image"], width=CASE_W - 30)
        h_before, w_before, _ = img_before.shape
        canvas[current_y : current_y + h_before, content_x : content_x + w_before] = img_before
        current_y += h_before + 25
        ocr_before_text = f"OCR: {case['original_ocr']} (Conf: {case['original_conf']:.2f})"
        cv2.putText(canvas, ocr_before_text, (content_x, current_y), FONT, 0.7, COLORS["fail"], 2)
        current_y += 35

        # Super-Resolved Image and its OCR Result
        img_after = resize_to_width(case["sr_image"], width=CASE_W - 30)
        h_after, w_after, _ = img_after.shape
        canvas[current_y : current_y + h_after, content_x : content_x + w_after] = img_after
        current_y += h_after + 25
        ocr_after_text = f"OCR: {case['sr_ocr']} (Conf: {case['sr_conf']:.2f})"
        cv2.putText(canvas, ocr_after_text, (content_x, current_y), FONT, 0.7, COLORS["success"], 2)

    cv2.imwrite(output_path, canvas)
    logger.info(f"Visualization panel saved to: {output_path}")



# --- Main Logic ---

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the script."""
    parser = argparse.ArgumentParser(description="Find and visualize convincing SR model improvements based on confidence.")
    parser.add_argument("--eval-folder", type=str, required=True, help="Path to the folder with evaluation images.")
    parser.add_argument("--sr-weights", type=str, required=True, help="Path to the Super-Resolution model weights (.pth).")
    parser.add_argument("--ocr-weights", type=str, required=True, help="Path to the OCR model weights (.pt).")
    parser.add_argument("--output-image", type=str, default="sr_convincing_improvements.png", help="Path to save the final visualization image.")
    parser.add_argument("--imgsz-ocr", nargs="+", type=int, default=[128, 128], help="OCR model inference size (h, w).")
    parser.add_argument("--ocr-conf", type=float, default=0.3, help="OCR confidence threshold for individual characters.")
    parser.add_argument("--iou", type=float, default=0.3, help="NMS IoU threshold for OCR.")
    parser.add_argument("--low-conf-thres", type=float, default=0.45, help="Average confidence below which original OCR is considered 'unreliable'.")
    parser.add_argument("--high-conf-thres", type=float, default=0.7, help="Average confidence above which SR OCR is considered 'reliable'.")
    parser.add_argument("--device", default="cuda", help="Computation device: 'cuda' or 'cpu'.")
    return parser.parse_args()

def main(args: argparse.Namespace):
    """Main function to find and visualize convincing improvement cases."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Models
    logger.info("Loading Super-Resolution model...")
    sr_model = LPSR(num_channels=3, num_features=32, growth_rate=16, num_blocks=4, num_layers=4, scale_factor=None).to(device)
    checkpoint = torch.load(args.sr_weights, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    sr_model.load_state_dict(state_dict)
    sr_model.eval()

    logger.info("Loading OCR model...")
    ocr_model = Detection(size=args.imgsz_ocr, weights_path=args.ocr_weights, device=device, iou_thres=args.iou, conf_thres=args.ocr_conf)

    # Data Preparation
    image_files = [f for f in os.listdir(args.eval_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(image_files)
    if not image_files:
        logger.error(f"No images found in evaluation folder: {args.eval_folder}")
        return
    
    # --- UPDATED: Search for 6 cases instead of 5 ---
    num_cases_to_find = 6
    logger.info(f"Searching for {num_cases_to_find} convincing improvement cases from {len(image_files)} images...")

    # Search Loop
    improvement_cases = []
    
    for filename in image_files:
        image_path = os.path.join(args.eval_folder, filename)
        ground_truth = get_ground_truth_from_filename(filename)
        
        original_image = cv2.imread(image_path)
        if original_image is None:
            continue

        original_ocr_text, original_avg_conf = run_ocr(ocr_model, original_image.copy())

        with torch.no_grad():
            sr_input_tensor = preprocess_for_sr(original_image).to(device)
            sr_output_tensor = sr_model(sr_input_tensor).squeeze(0).cpu()
            sr_np = sr_output_tensor.permute(1, 2, 0).numpy()
            sr_np = np.clip(sr_np, 0, 1)
            sr_image_bgr = cv2.cvtColor((sr_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        sr_ocr_text, sr_avg_conf = run_ocr(ocr_model, sr_image_bgr.copy())
        
        is_unreliable_before = original_avg_conf < args.low_conf_thres
        is_reliable_and_correct_after = (sr_ocr_text == ground_truth) and (sr_avg_conf >= args.high_conf_thres)

        if is_unreliable_before and is_reliable_and_correct_after:
            logger.success(f"Found convincing case {len(improvement_cases)+1}/{num_cases_to_find}: {filename}")
            improvement_cases.append({
                "original_image": original_image,
                "sr_image": sr_image_bgr,
                "ground_truth": ground_truth,
                "original_ocr": original_ocr_text,
                "original_conf": original_avg_conf,
                "sr_ocr": sr_ocr_text,
                "sr_conf": sr_avg_conf,
            })

        # --- UPDATED: Stop when enough cases are found ---
        if len(improvement_cases) >= num_cases_to_find:
            logger.info(f"Found {num_cases_to_find} convincing improvement cases. Proceeding to visualization.")
            break

    # Final Visualization
    if len(improvement_cases) > 0:
        create_visualization_panel(improvement_cases, args.output_image)
    else:
        logger.warning("Could not find any cases matching the specified criteria after searching all images.")

    logger.info("Script finished.")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)