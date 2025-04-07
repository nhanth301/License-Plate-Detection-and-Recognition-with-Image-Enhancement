import argparse
import os
import cv2
import sys
sys.path.append(os.path.abspath('../models'))
from detection import Detection

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='object.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default='ch_imgs', help='Input dir')
    parser.add_argument('--des', type=str, default='ch_out', help='Ouput dir')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1280], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    return opt




if __name__ == '__main__':
    opt = parse_opt()
    if not os.path.exists(opt.des):
            os.makedirs(opt.des)
    char_model=Detection(size=opt.imgsz,weights_path=opt.weights,device=opt.device,iou_thres=opt.iou_thres,conf_thres=opt.conf_thres)
    path=opt.source

    img_names=os.listdir(path)

    for img_name in img_names:
        img=cv2.imread(os.path.join(path,img_name))
        results, resized_img=char_model.detect(img.copy())
        for name,conf,box in results:
            resized_img=cv2.putText(resized_img, "{}".format(name), (int(box[0]), int(box[1])-3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 0, 255), 2)
            resized_img = cv2.rectangle(resized_img, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,0,255), 1)
        cv2.imwrite(os.path.join(opt.des, img_name), resized_img)