import argparse
import os
import sys
import cv2
import torch
from loguru import logger

sys.path.append(os.path.abspath('../models'))
from detection import Detection


def parse_opt():
    """
    Parses command-line arguments for the license plate and character detection script.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments with attributes for each defined argument.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='char', help='Type of model: char or object.')
    parser.add_argument('--d-weights', nargs='+', type=str, default='object.pt',
                        help='Detection model path(s) or Triton URL(s).')
    parser.add_argument('--r-weights', nargs='+', type=str, default='char.pt',
                        help='Recognition model path(s) or Triton URL(s).')
    parser.add_argument('--source', type=str, default='fg_imgs', help='Source image file or directory.')
    parser.add_argument('--des', type=str, default='out', help='Output directory for annotated images.')
    parser.add_argument('--fgimgsz', nargs='+', type=int, default=[1280], help='Detection input size (h, w).')
    parser.add_argument('--lpimgsz', nargs='+', type=int, default=[128], help='Recognition input size (h, w).')
    parser.add_argument('--d-conf-thres', type=float, default=0.7, help='Detection confidence threshold.')
    parser.add_argument('--r-conf-thres', type=float, default=0.1, help='Recognition confidence threshold.')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='NMS IoU threshold.')
    parser.add_argument('--max-det', type=int, default=1000, help='Maximum detections per image.')
    parser.add_argument('--device', default='cpu',
                        help='Device to run models on (e.g., "cpu", "cuda", or "cuda:0").')

    opt = parser.parse_args()

    # Expand size to (h, w) if only one value is provided
    if len(opt.lpimgsz) == 1:
        opt.lpimgsz *= 2
    if len(opt.fgimgsz) == 1:
        opt.fgimgsz *= 2

    return opt


if __name__ == '__main__':
    opt = parse_opt()

    os.makedirs(opt.des, exist_ok=True)
    os.makedirs('plates', exist_ok=True)

    image_folder = opt.source
    image_names = os.listdir(image_folder)

    plate_model = Detection(
        size=opt.fgimgsz,
        weights_path=opt.d_weights,
        device=opt.device,
        iou_thres=opt.iou_thres,
        conf_thres=opt.d_conf_thres
    )

    char_model = Detection(
        size=opt.lpimgsz,
        weights_path=opt.r_weights,
        device=opt.device,
        iou_thres=opt.iou_thres,
        conf_thres=opt.r_conf_thres
    )

    for image_name in image_names:
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)

        if image is None:
            logger.warning(f"Failed to load image: {image_name}")
            continue

        plate_results, resized_image = plate_model.detect(image.copy(), bb_scale=True)

        for i, (label, confidence, bbox) in enumerate(plate_results):
            if 'license plate' not in label.lower():
                continue

            x1, y1, x2, y2 = map(int, bbox)
            logger.info(
                f"Plate coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}, size=({x2 - x1}, {y2 - y1})"
            )

            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
            image = cv2.putText(
                image, "License Plate", (x1, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2
            )

            plate_crop = image[y1:y2, x1:x2]
            plate_crop_path = os.path.join('plates', f'plate_{image_name}_{i}.jpg')
            cv2.imwrite(plate_crop_path, plate_crop)
            logger.success(f"Saved cropped license plate to {plate_crop_path}")

            plate_resized = char_model.ResizeImg(plate_crop, size=(128, 128))
            char_results, _ = char_model.detect(plate_resized.copy())

            all_chars = ""

            if char_results:
                # Sort characters top-to-bottom by y1
                char_sorted = sorted(char_results, key=lambda x: x[2][1])

                rows = []
                current_row = []
                prev_y1 = char_sorted[0][2][1]

                for char_name, char_conf, char_box in char_sorted:
                    y1_char = char_box[1]
                    if abs(y1_char - prev_y1) > 20:
                        rows.append(current_row)
                        current_row = []
                    current_row.append((char_name, char_conf, char_box))
                    prev_y1 = y1_char
                if current_row:
                    rows.append(current_row)

                for row in rows:
                    row_sorted = sorted(row, key=lambda x: x[2][0])
                    row_groups = []
                    current_group = []
                    prev_x1 = row_sorted[0][2][0]

                    for char_name, char_conf, char_box in row_sorted:
                        x1_char = char_box[0]
                        if abs(x1_char - prev_x1) > 15:
                            row_groups.append("".join(c[0].upper() for c in current_group))
                            current_group = []
                        current_group.append((char_name, char_conf, char_box))
                        prev_x1 = x1_char

                    if current_group:
                        row_groups.append("".join(c[0].upper() for c in current_group))

                    all_chars += "".join(row_groups)

                logger.info(f"Detected license plate characters: {all_chars}")

                text_y = max(y1 - 20, 20)
                logger.debug(f"Drawing text at: ({x1}, {text_y})")

                image = cv2.putText(
                    image, all_chars, (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
                )
            else:
                logger.warning(f"No characters detected in license plate from {image_name}")

        output_path = os.path.join(opt.des, image_name)
        cv2.imwrite(output_path, image)
        logger.success(f"Saved final image with annotations to {output_path}")
