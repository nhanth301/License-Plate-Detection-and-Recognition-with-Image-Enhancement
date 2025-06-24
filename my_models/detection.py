import sys
import os
import cv2
import numpy as np
import torch

sys.path.append(os.path.abspath("./yolov5"))
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords


class Detection:
    def __init__(
        self,
        weights_path=".pt",
        size=(640, 640),
        device="cpu",
        iou_thres=None,
        conf_thres=None,
    ):
        self.device = device
        self.char_model, self.names = self.load_model(weights_path)
        self.size = size
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres

    def detect(self, frame, bb_scale=False):
        results, resized_img = self.char_detection_yolo(frame, bb_scale=bb_scale)
        return results, resized_img

    def preprocess_image(self, original_image):
        resized_img = self.ResizeImg(original_image, size=self.size)
        image = resized_img.copy()[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).to(self.device).float() / 255.0
        if image.ndimension() == 3:
            image = image.unsqueeze(0)
        return image, resized_img

    def char_detection_yolo(
        self, image, classes=None, agnostic_nms=True, max_det=1000, bb_scale=False
    ):
        img, resized_img = self.preprocess_image(image.copy())
        pred = self.char_model(img, augment=False)[0]
        detections = non_max_suppression(
            pred,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            classes=classes,
            agnostic=agnostic_nms,
            multi_label=True,
            labels=(),
            max_det=max_det,
        )

        results = []
        for i, det in enumerate(detections):
            if bb_scale:
                det[:, :4] = scale_coords(
                    resized_img.shape, det[:, :4], image.shape
                ).round()
            det = det.tolist()
            if len(det):
                for *xyxy, conf, cls in det:
                    result = [
                        self.names[int(cls)],
                        str(conf),
                        (xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                    ]
                    results.append(result)
        return results, resized_img

    def ResizeImg(self, img, size):
        h1, w1, _ = img.shape
        h, w = size
        if w1 < h1 * (w / h):
            new_w = int(float(w1 / h1) * h)
            img_rs = cv2.resize(img, (new_w, h))
            mask = np.zeros((h, w - new_w, 3), np.uint8)
            img = cv2.hconcat([img_rs, mask])
            trans_x = int(w / 2) - int(new_w / 2)
            trans_m = np.float32([[1, 0, trans_x], [0, 1, 0]])
        else:
            new_h = int(float(h1 / w1) * w)
            img_rs = cv2.resize(img, (w, new_h))
            mask = np.zeros((h - new_h, w, 3), np.uint8)
            img = cv2.vconcat([img_rs, mask])
            trans_y = int(h / 2) - int(new_h / 2)
            trans_m = np.float32([[1, 0, 0], [0, 1, trans_y]])

        height, width = img.shape[:2]
        img = cv2.warpAffine(img, trans_m, (width, height))
        return img

    def load_model(self, path, train=False):
        model = attempt_load(path, map_location=self.device)
        names = model.module.names if hasattr(model, "module") else model.names
        model.train() if train else model.eval()
        return model, names