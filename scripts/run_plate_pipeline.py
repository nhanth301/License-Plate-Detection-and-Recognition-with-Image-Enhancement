import argparse
import os
import cv2
import sys
import torch
from loguru import logger
sys.path.append(os.path.abspath('../models'))
from detection import Detection

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='char', help='type of model')
    parser.add_argument('--d-weights', nargs='+', type=str, default='object.pt', help='model path or triton URL')
    parser.add_argument('--r-weights', nargs='+', type=str, default='char.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default='fg_imgs', help='file/dir')
    parser.add_argument('--des', type=str, default='out', help='Output directory')
    parser.add_argument('--fgimgsz', nargs='+', type=int, default=[1280], help='inference size h,w')
    parser.add_argument('--lpimgsz', nargs='+', type=int, default=[128], help='inference size h,w')
    parser.add_argument('--d-conf-thres', type=float, default=0.7, help='confidence threshold')
    parser.add_argument('--r-conf-thres', type=float, default=0.1, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.lpimgsz *= 2 if len(opt.lpimgsz) == 1 else 1
    opt.fgimgsz *= 2 if len(opt.fgimgsz) == 1 else 1
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    if not os.path.exists(opt.des):
        os.makedirs(opt.des)
    if not os.path.exists('plates'):
        os.makedirs('plates')

    path = opt.source
    img_names = os.listdir(path)

    plate_model = Detection(size=opt.fgimgsz, weights_path=opt.d_weights, device=opt.device,
                            iou_thres=opt.iou_thres, conf_thres=opt.d_conf_thres)
    char_model = Detection(size=opt.lpimgsz, weights_path=opt.r_weights, device=opt.device,
                           iou_thres=opt.iou_thres, conf_thres=opt.r_conf_thres)

    for img_name in img_names:
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Failed to load image: {img_name}")
            continue

        plate_results, resized_img = plate_model.detect(img.copy(), bb_scale=True)
        for i, (plate_name, plate_conf, plate_box) in enumerate(plate_results):
            if plate_name != 'square license plate':
                continue

            x1, y1, x2, y2 = map(int, plate_box)
            logger.info(f"Plate coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}, size=({x2 - x1}, {y2 - y1})")

            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            img = cv2.putText(img, "License Plate", (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            plate_img = img[y1:y2, x1:x2]
            plate_img_path = os.path.join('plates', f'plate_{img_name}_{i}.jpg')
            cv2.imwrite(plate_img_path, plate_img)
            logger.success(f"Saved cropped license plate to {plate_img_path}")

            plate_img_resized = char_model.ResizeImg(plate_img, size=(128, 128))
            char_results, _ = char_model.detect(plate_img_resized.copy())

            all_chars = ""
            if char_results:
                char_results_sorted = sorted(char_results, key=lambda x: x[2][1])

                rows = []
                current_row = []
                prev_y1 = char_results_sorted[0][2][1]
                for char_name, char_conf, char_box in char_results_sorted:
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
                    row_str = "".join(row_groups)
                    all_chars += row_str

                logger.info(f"Detected license plate characters: {all_chars}")

                text_y = max(y1 - 20, 20)
                logger.debug(f"Drawing text at: ({x1}, {text_y})")

                img = cv2.putText(img, all_chars, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            else:
                logger.warning(f"No characters detected in license plate from {img_name}")

        output_path = os.path.join(opt.des, img_name)
        cv2.imwrite(output_path, img)
        logger.success(f"Saved final image with annotations to {output_path}")