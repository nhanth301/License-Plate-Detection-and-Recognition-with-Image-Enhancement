import argparse
import os
import cv2
import sys
from loguru import logger

sys.path.append(os.path.abspath('../models'))
sys.path.append(os.path.abspath('../utils'))
from detection import Detection
from lp_utils import sort_license_plate_detections

def parse_opt():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with configuration for detection.
    """
    parser = argparse.ArgumentParser(description="Character recognition from license plate images")
    parser.add_argument('--weights', nargs='+', type=str, default='char.pt', help='Model path or triton URL')
    parser.add_argument('--source', type=str, default='ch_imgs', help='Input image directory')
    parser.add_argument('--des', type=str, default='ch_out', help='Output image directory')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[128], help='Inference image size (h, w)')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='Maximum detections per image')
    parser.add_argument('--device', default='cpu', help='CUDA device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--keep-size', action='store_true', help='Keep original image size in output')
    parser.add_argument('--test', action='store_true', help='Run in test mode to compute accuracy metrics')
    opt = parser.parse_args()

    if len(opt.imgsz) == 1:
        opt.imgsz *= 2

    return opt

def is_image_file(filename):
    """
    Check if a file is an image based on its extension.

    Parameters
    ----------
    filename : str
        Name of the file to check.

    Returns
    -------
    bool
        True if the file is an image, False otherwise.
    """
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))

def main():
    """
    Main function for running character detection on license plate images.
    It handles both inference and test mode for accuracy evaluation.
    """
    opt = parse_opt()

    detector = Detection(
        size=opt.imgsz,
        weights_path=opt.weights,
        device=opt.device,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres
    )

    if opt.test:
        total_chars = correct_chars = correct_plates = total_plates = 0

        for img_name in os.listdir(opt.source):
            if not is_image_file(img_name):
                continue

            img_path = os.path.join(opt.source, img_name)
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Cannot read image: {img_path}")
                continue

            results, resized_img = detector.detect(img.copy(), bb_scale=opt.keep_size)
            if opt.keep_size:
                resized_img = img

            detected_text = ""
            if results:
                results_sorted = sort_license_plate_detections(results)

                for name, _, box in results_sorted:
                    label = name.upper()
                    detected_text += label
                    x1, y1, x2, y2 = map(int, box)

                    cv2.rectangle(resized_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    font_scale, thickness = 0.5, 1
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    lx, ly = x1, y1 - 5 if y1 - th - 5 >= 0 else y2 + th + 5

                    cv2.rectangle(resized_img, (lx, ly - th - 2), (lx + tw, ly), (0, 0, 0), -1)
                    cv2.putText(resized_img, label, (lx, ly - 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)

                cv2.putText(resized_img, detected_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (127, 255, 0), 1)
                gt = os.path.splitext(img_name)[0].upper()
                match_count = sum(1 for d, g in zip(detected_text, gt) if d == g)
                correct_chars += match_count
                total_chars += max(len(gt), len(detected_text))
                if detected_text == gt:
                    correct_plates += 1
                total_plates += 1

                logger.debug(f"GT: {gt} | Pred: {detected_text}")

                if total_plates == 100:
                    break
            else:
                logger.debug(f"{img_name} --> No characters detected")

        logger.info(f"✅ Character Accuracy: {correct_chars}/{total_chars} ({correct_chars / total_chars * 100:.2f}%)")
        logger.info(f"✅ Plate Accuracy: {correct_plates}/{total_plates} ({correct_plates / total_plates * 100:.2f}%)")

    else:
        os.makedirs(opt.des, exist_ok=True)
        for img_name in os.listdir(opt.source):
            if not is_image_file(img_name):
                continue

            img_path = os.path.join(opt.source, img_name)
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Cannot read image: {img_path}")
                continue

            results, resized_img = detector.detect(img.copy(), bb_scale=opt.keep_size)
            if opt.keep_size:
                resized_img = img

            detected_text = ""
            if results:
                results_sorted = sort_license_plate_detections(results)

                for name, _, box in results_sorted:
                    label = name.upper()
                    detected_text += label
                    x1, y1, x2, y2 = map(int, box)

                    cv2.rectangle(resized_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    font_scale, thickness = 0.5, 1
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    lx, ly = x1, y1 - 5 if y1 - th - 5 >= 0 else y2 + th + 5

                    cv2.rectangle(resized_img, (lx, ly - th - 2), (lx + tw, ly), (0, 0, 0), -1)
                    cv2.putText(resized_img, label, (lx, ly - 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)

                logger.debug(f"{img_name} --> {detected_text}")
                cv2.putText(resized_img, detected_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (127, 255, 0), 1)
            else:
                logger.debug(f"{img_name} --> No characters detected")

            cv2.imwrite(os.path.join(opt.des, img_name), resized_img)

if __name__ == '__main__':
    main()