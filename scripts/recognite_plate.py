import argparse
import os
import cv2
import sys
from loguru import logger

sys.path.append(os.path.abspath('../models'))
from detection import Detection

def parse_opt():
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
    opt = parser.parse_args()
    if len(opt.imgsz) == 1:
        opt.imgsz = opt.imgsz * 2
    return opt

def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))

def main():
    opt = parse_opt()
    os.makedirs(opt.des, exist_ok=True)

    detector = Detection(
        size=opt.imgsz,
        weights_path=opt.weights,
        device=opt.device,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres
    )

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
            results_sorted = sorted(results, key=lambda x: x[2][0])
            for name, conf, box in results_sorted:
                label = name.upper()
                detected_text += label
                x1, y1, x2, y2 = map(int, box)

                # Draw bounding box
                cv2.rectangle(resized_img, (x1, y1), (x2, y2), (0, 0, 255), 1)

                # Draw label
                font_scale, thickness = 0.5, 1
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                lx, ly = x1, y1 - 5 if y1 - th - 5 >= 0 else y2 + th + 5

                cv2.rectangle(resized_img, (lx, ly - th - 2), (lx + tw, ly), (0, 0, 0), -1)
                cv2.putText(resized_img, label, (lx, ly - 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)

            logger.info(f"{img_name} --> {detected_text}")
            cv2.putText(resized_img, detected_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (127, 255, 0), 1)
        else:
            logger.info(f"{img_name} --> No characters detected")

        cv2.imwrite(os.path.join(opt.des, img_name), resized_img)

if __name__ == '__main__':
    main()
