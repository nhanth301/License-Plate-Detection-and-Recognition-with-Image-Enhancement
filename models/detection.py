import numpy as np
import torch
import os
import sys
sys.path.append(os.path.abspath('../yolov5'))
from utils.general import non_max_suppression, scale_coords
from models.experimental import attempt_load
import cv2

class Detection:
    def __init__(self, weights_path='.pt', size=(640, 640), device='cpu', iou_thres=None, conf_thres=None):
        self.device = device
        self.char_model, self.names = self.load_model(weights_path)
        self.size = size
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres

    def detect(self, frame):
        results, resized_img = self.char_detection_yolo(frame)
        return results, resized_img

    def preprocess_image(self, original_image):
        resized_img = self.ResizeImg(original_image, size=self.size)
        image = resized_img.copy()[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).to(self.device)
        image = image.float()
        image /= 255.0
        if image.ndimension() == 3:
            image = image.unsqueeze(0)
        return image, resized_img

    def char_detection_yolo(self, image, classes=None, agnostic_nms=True, max_det=1000):
        img, resized_img = self.preprocess_image(image.copy())
        pred = self.char_model(img, augment=False)[0]
        detections = non_max_suppression(pred, conf_thres=self.conf_thres,
                                         iou_thres=self.iou_thres,
                                         classes=classes,
                                         agnostic=agnostic_nms,
                                         multi_label=True,
                                         labels=(),
                                         max_det=max_det)
        results = []
        for i, det in enumerate(detections):
            det = det.tolist()
            if len(det):
                for *xyxy, conf, cls in det:
                    result = [self.names[int(cls)], str(conf), (xyxy[0], xyxy[1], xyxy[2], xyxy[3])]
                    results.append(result)
        return results, resized_img

    def ResizeImg(self, img, size):
        h1, w1, _ = img.shape
        h, w = size
        if w1 < h1 * (w / h):
            img_rs = cv2.resize(img, (int(float(w1 / h1) * h), h))
            mask = np.zeros((h, w - int(float(w1 / h1) * h), 3), np.uint8)
            img = cv2.hconcat([img_rs, mask])
            trans_x = int(w / 2) - int(int(float(w1 / h1) * h) / 2)
            trans_y = 0
            trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
            height, width = img.shape[:2]
            img = cv2.warpAffine(img, trans_m, (width, height))
            return img
        else:
            img_rs = cv2.resize(img, (w, int(float(h1 / w1) * w)))
            mask = np.zeros((h - int(float(h1 / w1) * w), w, 3), np.uint8)
            img = cv2.vconcat([img_rs, mask])
            trans_x = 0
            trans_y = int(h / 2) - int(int(float(h1 / w1) * w) / 2)
            trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
            height, width = img.shape[:2]
            img = cv2.warpAffine(img, trans_m, (width, height))
            return img

    def load_model(self, path, train=False):
        model = attempt_load(path, map_location=self.device)
        names = model.module.names if hasattr(model, 'module') else model.names
        if train:
            model.train()
        else:
            model.eval()
        return model, names