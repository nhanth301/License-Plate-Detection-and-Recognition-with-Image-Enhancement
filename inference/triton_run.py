# file: run_pipeline_triton.py

import argparse
import os
import sys
import time
from typing import List, Tuple, Any

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image

import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import torchvision.transforms as transforms

# --- Assumed imports from your project structure ---
# Make sure these paths are correct for your project
try:
    # Adding yolov5 to path to find its utils
    sys.path.append(os.path.abspath('./yolov5'))
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression

    # Assuming your own utils are in a discoverable path, e.g., in a 'my_utils' folder
    from my_utils.utils import sort_license_plate_detections, straighten_license_plate
except ImportError as e:
    logger.error(f"Import Error: {e}. Please ensure yolov5 and my_utils are in the project structure.")
    sys.exit(1)


# --- File I/O and Image Utilities ---

def load_class_names(file_path: str) -> List[str]:
    """Loads class names from a text file (one name per line)."""
    try:
        with open(file_path, "r") as f:
            class_names = [line.strip() for line in f.readlines()]
        logger.info(f"Successfully loaded {len(class_names)} classes from {file_path}")
        return class_names
    except FileNotFoundError:
        logger.error(f"Class name file not found at: {file_path}")
        sys.exit(1)


def format_long_plate(plate_image: np.ndarray, aspect_ratio_threshold: float = 1.5) -> Tuple[np.ndarray, bool]:
    """Converts a 2-row license plate image into a single horizontal row."""
    h, w = plate_image.shape[:2]
    if h == 0 or w == 0:
        return plate_image, False

    if (w / h) > aspect_ratio_threshold:
        return plate_image, False

    mid_y = h // 2
    top_half = plate_image[0:mid_y, :]
    bottom_half = plate_image[h - mid_y : h, :]

    if top_half.shape[1] != bottom_half.shape[1]:
        min_w = min(top_half.shape[1], bottom_half.shape[1])
        top_half = top_half[:, :min_w]
        bottom_half = bottom_half[:, :min_w]

    return cv2.hconcat([top_half, bottom_half]), True


def restack_to_square(long_plate_image: np.ndarray, aspect_ratio_threshold: float = 1.5) -> np.ndarray:
    """Converts a long (1-row) license plate image back to a square (2-row) format."""
    h, w = long_plate_image.shape[:2]
    if h == 0 or w == 0 or (w / h) < aspect_ratio_threshold:
        return long_plate_image

    mid_x = w // 2
    left_half = long_plate_image[:, 0:mid_x]
    right_half = long_plate_image[:, w - mid_x : w]

    return cv2.vconcat([left_half, right_half])


# --- Triton Inference Functions ---

def run_sr_inference(
    triton_client: httpclient.InferenceServerClient,
    model_name: str,
    image: np.ndarray,
    target_size: Tuple[int, int] = (192, 32),
) -> np.ndarray:
    """Sends an image to the Super-Resolution model on Triton."""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose([
        transforms.Resize(size=(target_size[1], target_size[0])),
        transforms.ToTensor(),
    ])
    input_tensor = transform(pil_image).unsqueeze(0)
    input_numpy = input_tensor.numpy()

    infer_input = httpclient.InferInput("input_image", input_numpy.shape, "FP32")
    infer_input.set_data_from_numpy(input_numpy, binary_data=False)
    infer_output = httpclient.InferRequestedOutput("output_image", binary_data=False)

    response = triton_client.infer(model_name=model_name, inputs=[infer_input], outputs=[infer_output])
    sr_numpy = response.as_numpy("output_image")
    
    sr_numpy = np.squeeze(sr_numpy, axis=0)
    sr_numpy = (sr_numpy + 1) / 2.0  # Denormalize from [-1, 1] to [0, 1] if needed
    sr_numpy = np.clip(sr_numpy, 0, 1)
    sr_numpy_hwc = np.transpose(sr_numpy, (1, 2, 0))
    sr_bgr = cv2.cvtColor((sr_numpy_hwc * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    return sr_bgr


def run_detection_inference(
    triton_client: httpclient.InferenceServerClient,
    model_name: str,
    class_names: List[str],
    image: np.ndarray,
    target_size: Tuple[int, int],
    conf_thres: float,
    iou_thres: float,
) -> List[List[Any]]:
    """Sends an image to a YOLO model on Triton, using letterbox preprocessing."""
    h_orig, w_orig = image.shape[:2]

    img, ratio, (dw, dh) = letterbox(image, new_shape=target_size, stride=32, auto=False)
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    input_tensor = np.expand_dims(img, 0).astype(np.float32) / 255.0

    infer_input = httpclient.InferInput("input_image", input_tensor.shape, "FP32")
    infer_input.set_data_from_numpy(input_tensor, binary_data=False)
    infer_output = httpclient.InferRequestedOutput("predictions", binary_data=False)
    
    response = triton_client.infer(model_name=model_name, inputs=[infer_input], outputs=[infer_output])
    predictions = torch.from_numpy(response.as_numpy("predictions"))

    detections = non_max_suppression(
        predictions, conf_thres=conf_thres, iou_thres=iou_thres
    )[0]

    results = []
    if detections is not None and len(detections):
        coords = detections[:, :4]
        coords[:, [0, 2]] -= dw
        coords[:, [1, 3]] -= dh
        coords[:, :4] /= min(ratio)
        
        coords[:, 0].clamp_(0, w_orig)
        coords[:, 1].clamp_(0, h_orig)
        coords[:, 2].clamp_(0, w_orig)
        coords[:, 3].clamp_(0, h_orig)

        for *xyxy, conf, cls in detections:
            box = tuple(map(int, xyxy))
            label = class_names[int(cls)]
            results.append([label, f"{conf:.2f}", box])
    return results



def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="End-to-End License Plate Recognition with Triton")
    parser.add_argument("--source", type=str, required=True, help="Path to input video file or image.")
    parser.add_argument("--triton-url", type=str, default="localhost:8000", help="URL of the Triton Inference Server.")
    parser.add_argument("--sr-model-name", type=str, default="sr", help="Name of the Super-Resolution model on Triton.")
    parser.add_argument("--detection-model-name", type=str, default="detection", help="Name of the plate detection model on Triton.")
    parser.add_argument("--ocr-model-name", type=str, default="ocr", help="Name of the OCR model on Triton.")
    parser.add_argument("--detection-classes", type=str, default="yolo_classes/detect_class_names.txt", help="Path to class names for detection model.")
    parser.add_argument("--ocr-classes", type=str, default="yolo_classes/ocr_class_names.txt", help="Path to class names for OCR model.")
    parser.add_argument("--display-size", nargs="+", type=int, default=[1400, 900], help="Output window size (W, H).")
    parser.add_argument("--detection-conf", type=float, default=0.4, help="Confidence threshold for plate detection.")
    parser.add_argument("--ocr-conf", type=float, default=0.25, help="Confidence threshold for character recognition.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    return parser.parse_args()


def main(args: argparse.Namespace):
    """Main function to run the video processing pipeline."""
    try:
        triton_client = httpclient.InferenceServerClient(url=args.triton_url, verbose=False)
        if not triton_client.is_server_live():
            logger.error(f"Triton server is not live at {args.triton_url}. Exiting.")
            return
    except Exception as e:
        logger.error(f"Could not create Triton client: {e}. Exiting.")
        return
    logger.info(f"Successfully connected to Triton server at: {args.triton_url}")

    plate_class_names = load_class_names(args.detection_classes)
    ocr_class_names = load_class_names(args.ocr_classes)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        logger.error(f"Cannot open video source: {args.source}")
        return

    prev_time = time.time()
    display_w, display_h = args.display_size
    COLORS = {
        "plate1": (0, 255, 128), "plate2": (0, 215, 255), "plate3": (255, 128, 0),
        "text": (255, 255, 255), "conf": (173, 216, 230), "fps": (144, 238, 144),
        "bg": (30, 30, 30)
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video stream reached.")
            break

        original_frame = frame.copy()
        display_frame = np.full((display_h, display_w, 3), COLORS["bg"], dtype=np.uint8)

        main_h, main_w = frame.shape[:2]
        main_aspect = main_w / main_h if main_h > 0 else 1
        main_display_h = int(display_h * 0.55)
        main_display_w = int(main_display_h * main_aspect)
        if main_display_w > display_w:
            main_display_w, main_display_h = display_w, int(display_w / main_aspect)
        main_resized = cv2.resize(frame, (main_display_w, main_display_h))
        y_offset, x_offset = 10, (display_w - main_display_w) // 2
        display_frame[y_offset : y_offset + main_display_h, x_offset : x_offset + main_display_w] = main_resized

        plate_results = run_detection_inference(
            triton_client, args.detection_model_name, plate_class_names, original_frame,
            target_size=(1280, 1280), conf_thres=args.detection_conf, iou_thres=args.iou
        )
        
        allowed_plate_classes = {"square license plate", "rectangle license plate"}
        license_plates = [p for p in plate_results if p[0].lower() in allowed_plate_classes]
        license_plates.sort(key=lambda x: (x[2][2] - x[2][0]) * (x[2][3] - x[2][1]), reverse=True)

        detected_plates_info = []
        for i, (plate_name, plate_conf, plate_box) in enumerate(license_plates[:3]):
            x1, y1, x2, y2 = plate_box
            if x1 < 0 or y1 < 0 or x2 > main_w or y2 > main_h: continue

            scale_x, scale_y = main_display_w / main_w, main_display_h / main_h
            scaled_box = [int(x1 * scale_x) + x_offset, int(y1 * scale_y) + y_offset, int(x2 * scale_x) + x_offset, int(y2 * scale_y) + y_offset]
            cv2.rectangle(display_frame, (scaled_box[0], scaled_box[1]), (scaled_box[2], scaled_box[3]), COLORS[f'plate{i+1}'], 2)
            cv2.putText(display_frame, f"#{i+1}", (scaled_box[0], scaled_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text'], 2)

            plate_img_raw = original_frame[y1:y2, x1:x2]
            if plate_img_raw.size == 0: continue
            
            plate_img_straight = straighten_license_plate(plate_img_raw)
            plate_img_long, was_formatted = format_long_plate(plate_img_straight)
            
            ocr_plate_input = restack_to_square(plate_img_long) if was_formatted else plate_img_long.copy()
            orig_char_results = run_detection_inference(
                triton_client, args.ocr_model_name, ocr_class_names, ocr_plate_input,
                target_size=(128, 128), conf_thres=args.ocr_conf, iou_thres=args.iou
            )
            orig_chars = "".join([res[0] for res in sort_license_plate_detections(orig_char_results)])

            hr_plate_img = run_sr_inference(triton_client, args.sr_model_name, plate_img_long)

            sr_plate_for_ocr = restack_to_square(hr_plate_img) if was_formatted else hr_plate_img.copy()
            sr_char_results = run_detection_inference(
                triton_client, args.ocr_model_name, ocr_class_names, sr_plate_for_ocr,
                target_size=(128, 128), conf_thres=args.ocr_conf, iou_thres=args.iou
            )
            sr_chars = "".join([res[0] for res in sort_license_plate_detections(sr_char_results)])
            
            detected_plates_info.append({
                'display_img': ocr_plate_input, 'sr_img': sr_plate_for_ocr,
                'orig_text': orig_chars.upper(), 'sr_text': sr_chars.upper(), 'confidence': plate_conf,
                'plate_num': i + 1
            })

        panel_y_start = main_display_h + 50
        panel_x_margin = 40
        available_width = display_w - (2 * panel_x_margin)
        slot_width = available_width // 3 if len(detected_plates_info) > 0 else 0
        
        for plate_info in detected_plates_info:
            i = plate_info['plate_num'] - 1
            color = COLORS[f"plate{plate_info['plate_num']}"]
            slot_x_start = panel_x_margin + (i * slot_width)
            display_img = plate_info['display_img']
            h_plate, w_plate = display_img.shape[:2]
            aspect_ratio = w_plate / h_plate if h_plate > 0 else 1.0
            fixed_plate_h = 100
            display_plate_w = int(fixed_plate_h * aspect_ratio)
            if display_plate_w > slot_width - 20:
                 display_plate_w = slot_width - 20
                 fixed_plate_h = int(display_plate_w / aspect_ratio)
            content_x_pos = slot_x_start + (slot_width - display_plate_w) // 2
            current_y = panel_y_start
            
            cv2.putText(display_frame, f"Plate #{plate_info['plate_num']} (Conf: {float(plate_info['confidence']):.2f})", (content_x_pos, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['conf'], 2)
            current_y += 25
            cv2.putText(display_frame, "Original", (content_x_pos, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1)
            current_y += 15
            if display_plate_w > 0 and fixed_plate_h > 0:
                orig_display = cv2.resize(display_img, (display_plate_w, fixed_plate_h))
                display_frame[current_y : current_y + fixed_plate_h, content_x_pos : content_x_pos + display_plate_w] = orig_display
            current_y += fixed_plate_h + 15
            cv2.putText(display_frame, "Super-Resolved", (content_x_pos, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1)
            current_y += 15
            if display_plate_w > 0 and fixed_plate_h > 0:
                sr_display = cv2.resize(plate_info['sr_img'], (display_plate_w, fixed_plate_h))
                display_frame[current_y : current_y + fixed_plate_h, content_x_pos : content_x_pos + display_plate_w] = sr_display
            current_y += fixed_plate_h + 35
            font, font_scale, font_thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
            cv2.putText(display_frame, f"OCR: {plate_info['orig_text']}", (content_x_pos, current_y), font, font_scale, (200, 200, 200), font_thickness)
            current_y += 35
            cv2.putText(display_frame, f"SR OCR: {plate_info['sr_text']}", (content_x_pos, current_y), font, font_scale, color, font_thickness)
        
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        cv2.rectangle(display_frame, (5, 5), (150, 40), COLORS['bg'], -1)
        cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS['fps'], 2)

        cv2.imshow("License Plate Recognition (Triton)", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Pipeline finished.")


if __name__ == '__main__':
    args = parse_arguments()
    main(args)