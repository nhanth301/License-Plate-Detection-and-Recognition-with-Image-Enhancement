import argparse
import os
import sys
import time
from typing import List, Tuple, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from PIL import Image
import torchvision.transforms as T

from my_models.detection import Detection
from my_models.lpsr import LPSR
from my_utils.utils import sort_license_plate_detections, straighten_license_plate


# --- Image Processing Utilities ---
def format_long_plate(
    plate_image: np.ndarray, aspect_ratio_threshold: float = 1.5
) -> Tuple[np.ndarray, bool]:
    """
    Converts a 2-row license plate image into a single horizontal row.
    If the image is already long, returns the original image.

    Args:
        plate_image (np.ndarray): The input plate image.
        aspect_ratio_threshold (float): The w/h ratio above which a plate is considered long.

    Returns:
        A tuple containing the formatted plate image and a boolean indicating if a format change occurred.
    """
    h, w = plate_image.shape[:2]
    if h == 0 or w == 0:
        return plate_image, False

    if (w / h) > aspect_ratio_threshold:
        return plate_image, False  # Already a long plate

    mid_y = h // 2
    top_half = plate_image[0:mid_y, :]
    bottom_half = plate_image[h - mid_y : h, :]

    # Ensure width is consistent before concatenation
    if top_half.shape[1] != bottom_half.shape[1]:
        min_w = min(top_half.shape[1], bottom_half.shape[1])
        top_half = top_half[:, :min_w]
        bottom_half = bottom_half[:, :min_w]

    return cv2.hconcat([top_half, bottom_half]), True


def restack_to_square(
    long_plate_image: np.ndarray, aspect_ratio_threshold: float = 1.5
) -> np.ndarray:
    """
    Converts a long (1-row) license plate image back to a square (2-row) format.
    If the image is already square, returns the original image.

    Args:
        long_plate_image (np.ndarray): The input long plate image.
        aspect_ratio_threshold (float): The w/h ratio below which a plate is considered square.

    Returns:
        The restacked square plate image.
    """
    h, w = long_plate_image.shape[:2]
    if h == 0 or w == 0 or (w / h) < aspect_ratio_threshold:
        return long_plate_image  # Already a square plate

    mid_x = w // 2
    left_half = long_plate_image[:, 0:mid_x]
    right_half = long_plate_image[:, w - mid_x : w]

    return cv2.vconcat([left_half, right_half])


def preprocess_for_sr(
    plate_image: np.ndarray, target_size: Tuple[int, int] = (192, 32)
) -> torch.Tensor:
    """
    Preprocesses a plate image for the Super-Resolution model.

    Args:
        plate_image (np.ndarray): Input plate image in BGR format.
        target_size (Tuple[int, int]): Target size (W, H) for the model.

    Returns:
        A preprocessed image tensor ready for the SR model.
    """
    rgb_img = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img).resize(target_size, Image.BICUBIC)
    transform = T.Compose([T.ToTensor()])
    return transform(pil_img).unsqueeze(0)


# --- Main Application Logic ---

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the application."""
    parser = argparse.ArgumentParser(description="License Plate Recognition System")
    parser.add_argument("--d-weights", type=str, required=True, help="Detection model weights path (.pt)")
    parser.add_argument("--r-weights", type=str, required=True, help="Recognition (OCR) model weights path (.pt)")
    parser.add_argument("--sr-weights", type=str, required=True, help="Super-resolution model weights path (.pth)")
    parser.add_argument("--source", type=str, required=True, help="Path to input video file or image source.")
    parser.add_argument("--imgsz-det", nargs="+", type=int, default=[1280, 1280], help="Detection inference size (h, w)")
    parser.add_argument("--imgsz-ocr", nargs="+", type=int, default=[128, 128], help="Recognition inference size (h, w)")
    parser.add_argument("--d-conf", type=float, default=0.7, help="Detection confidence threshold")
    parser.add_argument("--r-conf", type=float, default=0.25, help="Recognition confidence threshold")
    parser.add_argument("--iou", type=float, default=0.3, help="NMS IoU threshold")
    parser.add_argument("--device", default="cuda", help="Computation device: 'cuda' or 'cpu'")
    parser.add_argument("--display-size", nargs="+", type=int, default=[1400, 900], help="Output window size (w, h)")
    return parser.parse_args()


def main(opt: argparse.Namespace):
    """Main function to run the video processing and recognition pipeline."""
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load Models ---
    sr_model = LPSR(num_channels=3, num_features=32, growth_rate=16, num_blocks=4, num_layers=4, scale_factor=None).to(device)
    checkpoint = torch.load(opt.sr_weights, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    sr_model.load_state_dict(state_dict)
    sr_model.eval()
    logger.info("Super-Resolution model loaded.")

    plate_model = Detection(size=opt.imgsz_det, weights_path=opt.d_weights, device=device, iou_thres=opt.iou, conf_thres=opt.d_conf)
    logger.info("Plate Detection model loaded.")
    
    char_model = Detection(size=opt.imgsz_ocr, weights_path=opt.r_weights, device=device, iou_thres=opt.iou, conf_thres=opt.r_conf)
    logger.info("Character Recognition (OCR) model loaded.")

    # --- Video Capture ---
    cap = cv2.VideoCapture(opt.source)
    if not cap.isOpened():
        logger.error(f"Cannot open video source: {opt.source}")
        return

    # --- UI and Display Constants ---
    prev_time = time.time()
    display_w, display_h = opt.display_size
    COLORS = {
        "plate1": (0, 255, 128), "plate2": (0, 215, 255), "plate3": (255, 128, 0),
        "text": (255, 255, 255), "conf": (173, 216, 230), "fps": (144, 238, 144),
        "bg": (30, 30, 30)
    }

    # --- Main Video Loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video stream reached.")
            break

        original_frame = frame.copy()
        display_frame = np.full((display_h, display_w, 3), COLORS["bg"], dtype=np.uint8)

        # --- Main Frame Display Logic ---
        main_h, main_w = frame.shape[:2]
        main_aspect = main_w / main_h if main_h > 0 else 1
        main_display_h = int(display_h * 0.55)
        main_display_w = int(main_display_h * main_aspect)
        if main_display_w > display_w:
            main_display_w, main_display_h = display_w, int(display_w / main_aspect)
        
        main_resized = cv2.resize(frame, (main_display_w, main_display_h))
        y_offset, x_offset = 10, (display_w - main_display_w) // 2
        display_frame[y_offset : y_offset + main_display_h, x_offset : x_offset + main_display_w] = main_resized

        # --- Inference Pipeline ---
        plate_results, _ = plate_model.detect(original_frame.copy(), bb_scale=True)
        license_plates = [p for p in plate_results if "license plate" in p[0].lower()]
        license_plates.sort(key=lambda x: (x[2][2] - x[2][0]) * (x[2][3] - x[2][1]), reverse=True)

        detected_plates_info = []
        for i, (plate_name, plate_conf, plate_box) in enumerate(license_plates[:3]):
            x1, y1, x2, y2 = map(int, plate_box)

            scale_x, scale_y = main_display_w / main_w, main_display_h / main_h
            scaled_box = [int(x1 * scale_x) + x_offset, int(y1 * scale_y) + y_offset, int(x2 * scale_x) + x_offset, int(y2 * scale_y) + y_offset]
            cv2.rectangle(display_frame, (scaled_box[0], scaled_box[1]), (scaled_box[2], scaled_box[3]), COLORS[f"plate{i+1}"], 2)
            cv2.putText(display_frame, f"#{i+1}", (scaled_box[0], scaled_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["text"], 2)

            plate_img_raw = original_frame[y1:y2, x1:x2]
            if plate_img_raw.size == 0: continue

            plate_img_straight = straighten_license_plate(plate_img_raw)
            if plate_img_straight.size == 0: continue

            plate_img_long, was_formatted = format_long_plate(plate_img_straight)
            
            ocr_plate_input = restack_to_square(plate_img_long) if was_formatted else plate_img_long.copy()
            orig_char_results, _ = char_model.detect(ocr_plate_input)
            orig_chars = "".join([char_name.upper() for char_name, _, _ in sort_license_plate_detections(orig_char_results)])
            
            with torch.no_grad():
                sr_input_tensor = preprocess_for_sr(plate_img_long).to(device)
                sr_output_tensor = sr_model(sr_input_tensor).squeeze(0).cpu()
                sr_np = sr_output_tensor.permute(1, 2, 0).numpy() * 255
                hr_plate_img = cv2.cvtColor(sr_np.astype(np.uint8), cv2.COLOR_RGB2BGR)

            sr_plate_for_ocr = restack_to_square(hr_plate_img) if was_formatted else hr_plate_img.copy()
            sr_char_results, _ = char_model.detect(sr_plate_for_ocr)
            sr_chars = "".join([char_name.upper() for char_name, _, _ in sort_license_plate_detections(sr_char_results)])
            
            detected_plates_info.append({
                "display_img": ocr_plate_input, "sr_img": sr_plate_for_ocr,
                "orig_text": orig_chars, "sr_text": sr_chars, "confidence": plate_conf,
                "plate_num": i + 1,
            })

        # --- Display Panel Logic ---
        panel_y_start = main_display_h + 50
        panel_x_margin = 40
        available_width = display_w - (2 * panel_x_margin)
        slot_width = available_width // 3 if detected_plates_info else 0

        for plate_info in detected_plates_info:
            i = plate_info["plate_num"] - 1
            color = COLORS[f"plate{plate_info['plate_num']}"]
            slot_x_start = panel_x_margin + (i * slot_width)

            h_plate, w_plate = plate_info["display_img"].shape[:2]
            aspect_ratio = w_plate / h_plate if h_plate > 0 else 1.0
            
            fixed_plate_h = 100
            display_plate_w = int(fixed_plate_h * aspect_ratio)
            if display_plate_w > slot_width - 20:
                display_plate_w = slot_width - 20
                fixed_plate_h = int(display_plate_w / aspect_ratio)

            content_x_pos = slot_x_start + (slot_width - display_plate_w) // 2
            current_y = panel_y_start

            cv2.putText(display_frame, f"Detected Plate #{plate_info['plate_num']} (Conf: {float(plate_info['confidence']):.2f})", (content_x_pos, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["conf"], 2)
            current_y += 25

            cv2.putText(display_frame, "Original", (content_x_pos, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text"], 1)
            current_y += 15
            if display_plate_w > 0 and fixed_plate_h > 0:
                orig_display = cv2.resize(plate_info["display_img"], (display_plate_w, fixed_plate_h))
                display_frame[current_y : current_y + fixed_plate_h, content_x_pos : content_x_pos + display_plate_w] = orig_display
            current_y += fixed_plate_h + 15

            cv2.putText(display_frame, "Super-Resolved", (content_x_pos, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text"], 1)
            current_y += 15
            if display_plate_w > 0 and fixed_plate_h > 0:
                sr_display = cv2.resize(plate_info["sr_img"], (display_plate_w, fixed_plate_h))
                display_frame[current_y : current_y + fixed_plate_h, content_x_pos : content_x_pos + display_plate_w] = sr_display
            current_y += fixed_plate_h + 35

            font, font_scale, font_thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
            cv2.putText(display_frame, f"OCR: {plate_info['orig_text']}", (content_x_pos, current_y), font, font_scale, (200, 200, 200), font_thickness)
            current_y += 35
            cv2.putText(display_frame, f"SR OCR: {plate_info['sr_text']}", (content_x_pos, current_y), font, font_scale, color, font_thickness)

        # --- FPS Counter ---
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        cv2.rectangle(display_frame, (5, 5), (150, 40), COLORS["bg"], -1)
        cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS["fps"], 2)

        cv2.imshow("License Plate Recognition", display_frame)

        # Allow video to play. Press 'q' to quit.
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break
        key = cv2.waitKey(0) & 0xFF 
        if key == ord('q'):
            break
        elif key == 32:  
            continue  
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Pipeline finished.")


if __name__ == "__main__":
    opt = parse_arguments()
    main(opt)