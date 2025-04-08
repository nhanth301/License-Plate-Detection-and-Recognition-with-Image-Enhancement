import argparse
import os
import cv2
import sys
from loguru import logger
sys.path.append(os.path.abspath('../models'))
from detection import Detection

def parse_opt():
    parser = argparse.ArgumentParser(description="Detect license plates in input images")
    parser.add_argument('--weights', nargs='+', type=str, default='object.pt', help='Model path or Triton URL')
    parser.add_argument('--source', type=str, default='ch_imgs', help='Input directory')
    parser.add_argument('--des', type=str, default='ch_out', help='Output directory')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1280], help='Inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='Maximum detections per image')
    parser.add_argument('--device', default='cpu', help='CUDA device (e.g., 0 or cpu)')
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
            logger.warning(f"No image files found in {opt.source}")
            continue

        img_path = os.path.join(opt.source, img_name)
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Cannot read image: {img_path}")
            continue
        
        logger.debug(f"Running detection on image: {img_name}")
        results, resized_img = detector.detect(img.copy(), bb_scale=opt.keep_size)
        if opt.keep_size:
            resized_img = img

        for name, conf, box in results:
            x1, y1, x2, y2 = map(int, box)
            label = f"{name}"

            # Draw bounding box and label
            cv2.rectangle(resized_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(resized_img, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            logger.debug(f"Detected: {label} at [{x1}, {y1}, {x2}, {y2}]")

        output_path = os.path.join(opt.des, img_name)
        cv2.imwrite(output_path, resized_img)
        logger.info(f"Saved result to {output_path}")

if __name__ == '__main__':
    main()
