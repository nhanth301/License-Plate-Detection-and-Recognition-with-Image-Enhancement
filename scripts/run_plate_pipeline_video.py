import argparse
import os
import cv2
import sys
import time
import torch
from loguru import logger
import torchvision.transforms as T
from PIL import Image
import numpy as np

sys.path.append(os.path.abspath('../models'))
sys.path.append(os.path.abspath('../utils'))
from detection import Detection
from base_sp_lpr import LPSR
from lp_utils import sort_license_plate_detections, straighten_license_plate

# --- Image Processing Functions ---
def load_image_from_cv2(plate_img):
    """Convert OpenCV BGR image to a tensor for model input."""
    rgb_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img).resize((96, 32), Image.BICUBIC)
    transform = T.Compose([T.ToTensor()])
    return transform(pil_img).unsqueeze(0)

# --- Argument Parsing ---
def parse_opt():
    """Parse command-line arguments for configuration."""
    parser = argparse.ArgumentParser(description="License Plate Recognition System")
    parser.add_argument('--d-weights', nargs='+', type=str, default='object.pt', help='Detection model weights')
    parser.add_argument('--r-weights', nargs='+', type=str, default='char.pt', help='Recognition model weights')
    parser.add_argument('--source', type=str, default='video.mp4', help='Path to input video file')
    parser.add_argument('--fgimgsz', nargs='+', type=int, default=[1280], help='Foreground inference size (h,w)')
    parser.add_argument('--lpimgsz', nargs='+', type=int, default=[128], help='License plate inference size (h,w)')
    parser.add_argument('--d-conf-thres', type=float, default=0.7, help='Detection confidence threshold')
    parser.add_argument('--r-conf-thres', type=float, default=0.25, help='Recognition confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='Max detections per image')
    parser.add_argument('--device', default='cpu', help='Device: cuda or cpu')
    parser.add_argument('--display-size', nargs='+', type=int, default=[1400, 900], help='Output window size (w,h)')
    return parser.parse_args()

# --- Main Function ---
def main(opt):
    """Main function for license plate detection and recognition."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Model Initialization ---
    model = LPSR(num_channels=3, num_features=124, growth_rate=64, num_blocks=8, num_layers=4, scale_factor=2).to(device)
    checkpoint = torch.load('../weights/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    opt.lpimgsz = opt.lpimgsz * 2 if len(opt.lpimgsz) == 1 else opt.lpimgsz
    opt.fgimgsz = opt.fgimgsz * 2 if len(opt.fgimgsz) == 1 else opt.fgimgsz

    plate_model = Detection(size=opt.fgimgsz, weights_path=opt.d_weights, device=opt.device, iou_thres=opt.iou_thres, conf_thres=opt.d_conf_thres)
    char_model = Detection(size=opt.lpimgsz, weights_path=opt.r_weights, device=opt.device, iou_thres=opt.iou_thres, conf_thres=opt.r_conf_thres)

    # --- Video Setup ---
    cap = cv2.VideoCapture(opt.source)
    if not cap.isOpened():
        logger.error(f"Cannot open video file: {opt.source}")
        return

    prev_time = time.time()
    display_w, display_h = opt.display_size

    # Define color palette (BGR format)
    COLORS = {
        'plate1': (0, 255, 128),  # Bright green
        'plate2': (0, 215, 255),  # Gold
        'plate3': (255, 128, 0),  # Orange
        'text': (255, 255, 255),  # White
        'conf': (173, 216, 230),  # Light blue
        'fps': (144, 238, 144),  # Light green
        'bg': (30, 30, 30)       # Dark gray background
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video reached.")
            break

        original_frame = frame.copy()
        display_frame = np.full((display_h, display_w, 3), COLORS['bg'], dtype=np.uint8)

        # --- Resize and Position Main Video ---
        main_h, main_w = frame.shape[:2]
        main_aspect = main_w / main_h
        main_display_h = display_h // 2
        main_display_w = int(main_display_h * main_aspect)

        if main_display_w > display_w:
            main_display_w = display_w
            main_display_h = int(display_w / main_aspect)

        main_resized = cv2.resize(frame, (main_display_w, main_display_h))
        y_offset = 20
        x_offset = (display_w - main_display_w) // 2
        display_frame[y_offset:y_offset + main_display_h, x_offset:x_offset + main_display_w] = main_resized

        # --- License Plate Detection ---
        plate_results, _ = plate_model.detect(original_frame.copy(), bb_scale=True)

        def sort_plates_by_priority(plate_result):
            _, _, box = plate_result
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            return -area

        license_plates = [p for p in plate_results if 'license plate' in p[0].lower()]
        license_plates.sort(key=sort_plates_by_priority)

        detected_plates = []
        lp_x_start = 30
        lp_y_start = main_display_h + 60
        lp_height = (display_h - lp_y_start - 30) // 4  # Reduced height to fit two images per plate

        # --- Process Each Detected Plate ---
        for i, (plate_name, plate_conf, plate_box) in enumerate(license_plates):
            x1, y1, x2, y2 = map(int, plate_box)
            plate_num = i + 1

            # Draw bounding box and number on main video
            scale_x, scale_y = main_display_w / main_w, main_display_h / main_h
            scaled_box = [
                int(x1 * scale_x) + x_offset, int(y1 * scale_y) + y_offset,
                int(x2 * scale_x) + x_offset, int(y2 * scale_y) + y_offset
            ]
            color = COLORS[f'plate{i+1}' if i < 3 else 'plate3']
            cv2.rectangle(display_frame, (scaled_box[0], scaled_box[1]), (scaled_box[2], scaled_box[3]), color, 2)
            cv2.putText(display_frame, f"#{plate_num}", (scaled_box[0], scaled_box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text'], 2)

            # Extract original plate image
            plate_img = straighten_license_plate(original_frame[y1:y2, x1:x2])
            if plate_img.size == 0:
                continue

            # Character recognition on original plate image
            plate_img_resized = char_model.ResizeImg(plate_img, size=(128, 128))
            orig_char_results, _ = char_model.detect(plate_img_resized.copy())
            orig_chars = "".join([char_name.upper() for char_name, _, _ in sort_license_plate_detections(orig_char_results)])

            # Super-resolution enhancement
            with torch.no_grad():
                try:
                    sr_tensor = model(load_image_from_cv2(plate_img).to(device))
                    sr_tensor = sr_tensor.squeeze(0).cpu().clamp(0, 1)
                    sr_np = sr_tensor.permute(1, 2, 0).numpy() * 255
                    hr_plate_img = cv2.cvtColor(sr_np.astype('uint8'), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    logger.error(f"Super-resolution error: {e}")
                    hr_plate_img = plate_img

            # Character recognition on super-resolved image
            plate_img_resized = char_model.ResizeImg(hr_plate_img, size=(128, 128))
            sr_char_results, _ = char_model.detect(plate_img_resized.copy())
            sr_chars = "".join([char_name.upper() for char_name, _, _ in sort_license_plate_detections(sr_char_results)])

            detected_plates.append({
                'orig_img': plate_img,       # Original plate image
                'sr_img': hr_plate_img,      # Super-resolved plate image
                'orig_text': orig_chars,     # OCR result before super-resolution
                'sr_text': sr_chars,         # OCR result after super-resolution
                'confidence': plate_conf,
                'plate_num': plate_num
            })

        # --- Display Detected Plates ---
        plate_width = (display_w - 90) // 3  # Width for each plate column
        plate_spacing = 30

        for plate_info in detected_plates:
            if plate_info['plate_num'] > 3:
                continue

            i = plate_info['plate_num'] - 1
            x_pos = lp_x_start + i * (plate_width + plate_spacing)
            color = COLORS[f'plate{i+1}' if i < 3 else 'plate3']

            # --- Display Original Plate ---
            orig_img = plate_info['orig_img']
            h, w = orig_img.shape[:2]
            aspect_ratio = w / h
            display_plate_w = plate_width
            display_plate_h = int(display_plate_w / aspect_ratio)
            if display_plate_h > lp_height:
                display_plate_h = lp_height
                display_plate_w = int(display_plate_h * aspect_ratio)

            orig_display = cv2.resize(orig_img, (display_plate_w, display_plate_h))
            y_end_orig = min(lp_y_start + display_plate_h, display_h)
            x_end = min(x_pos + display_plate_w, display_w)

            if y_end_orig > lp_y_start and x_end > x_pos:
                try:
                    display_frame[lp_y_start:y_end_orig, x_pos:x_end] = orig_display[:y_end_orig-lp_y_start, :x_end-x_pos]
                except ValueError as e:
                    logger.error(f"Error displaying original plate: {e}")



            # --- Display Super-Resolved Plate ---
            sr_img = plate_info['sr_img']
            sr_display = cv2.resize(sr_img, (display_plate_w, display_plate_h))
            y_start_sr = lp_y_start + display_plate_h + 20  # Space between original and SR
            y_end_sr = min(y_start_sr + display_plate_h, display_h)

            if y_end_sr > y_start_sr and x_end > x_pos:
                try:
                    display_frame[y_start_sr:y_end_sr, x_pos:x_end] = sr_display[:y_end_sr-y_start_sr, :x_end-x_pos]
                except ValueError as e:
                    logger.error(f"Error displaying SR plate: {e}")

            # Label for super-resolved plate
            cv2.putText(display_frame, "Enhanced", (x_pos, y_start_sr - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['text'], 2)

            # --- Display Both OCR Texts and Confidence ---
            text_y = y_start_sr + display_plate_h + 40
            cv2.rectangle(display_frame, (x_pos - 5, lp_y_start - 35), (x_pos + 50, lp_y_start - 5), COLORS['bg'], -1)
            cv2.putText(display_frame, f"#{plate_info['plate_num']}", (x_pos, lp_y_start - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['text'], 2)
            cv2.putText(display_frame, f"Ori: {plate_info['orig_text']}", (x_pos, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(display_frame, f"SR: {plate_info['sr_text']}", (x_pos, text_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(display_frame, f"Conf: {float(plate_info['confidence']):.2f}", (x_pos, text_y + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS['conf'], 2)

        # --- FPS Display ---
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.rectangle(display_frame, (5, 5), (150, 40), COLORS['bg'], -1)
        cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS['fps'], 2)

        # --- Show Frame and Handle Input ---
        cv2.imshow("License Plate Recognition", display_frame)
        # key = cv2.waitKey(0) & 0xFF
        # if key == ord('q'):
        #     break
        # elif key == 32:  # Spacebar
        #     continue
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)