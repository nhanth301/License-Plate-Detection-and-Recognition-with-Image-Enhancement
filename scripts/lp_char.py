import argparse
import os
import cv2
import sys
sys.path.append(os.path.abspath('../models'))
from detection import Detection

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='char', help='Model type')
    parser.add_argument('--weights', nargs='+', type=str, default='char.pt', help='Model path or triton URL')
    parser.add_argument('--source', type=str, default='ch_imgs', help='Input dir')
    parser.add_argument('--des', type=str, default='ch_out', help='Ouput dir')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[128], help='Inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='Maximum detections per image')
    parser.add_argument('--device', default='cpu', help='CUDA device or CPU')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    return opt

def main():
    opt = parse_opt()
    if not os.path.exists(opt.des):
        os.makedirs(opt.des)
    detector = Detection(size=opt.imgsz, weights_path=opt.weights, device=opt.device,
                        iou_thres=opt.iou_thres, conf_thres=opt.conf_thres)
    
    for img_name in os.listdir(opt.source):
        img_path = os.path.join(opt.source, img_name)
        img = cv2.imread(img_path)
        results, resized_img = detector.detect(img.copy())
        
        detected_text = ""
        if results:
            results_sorted = sorted(results, key=lambda x: x[2][0])
            for name, conf, box in results_sorted:
                label = name.upper()
                detected_text += label
                x1, y1, x2, y2 = map(int, box)
                
                # Draw bounding box
                cv2.rectangle(resized_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
                
                # Configure label text
                font_scale, font_thickness = 0.5, 1
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                label_x = x1
                label_y = y1 - 5 if y1 - text_h - 5 >= 0 else y2 + text_h + 5
                
                # Draw label background and text
                cv2.rectangle(resized_img, (label_x, label_y - text_h - 2), 
                            (label_x + text_w, label_y), (0, 0, 0), -1)
                cv2.putText(resized_img, label, (label_x, label_y - 1), 
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), font_thickness)
            
            print(detected_text)
            cv2.putText(resized_img, detected_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, (127, 255, 0), 1)
        else:
            print(f"No characters detected in {img_name}")

        cv2.imwrite(os.path.join(opt.des, img_name), resized_img)

if __name__ == '__main__':
    main()