import numpy as np
import torch
import os
import sys
import cv2

# Add YOLOv5 module to the path
sys.path.append(os.path.abspath('../yolov5'))

from utils.general import non_max_suppression, scale_coords
from models.experimental import attempt_load


class Detection:
    def __init__(self, weights_path='.pt', size=(640, 640), device='cpu', iou_thres=None, conf_thres=None):
        """
        Initialize the Detection object for YOLO-based object detection.

        Args:
            weights_path (str): Path to the model weights (.pt file).
            size (tuple): Input size for the model as (height, width).
            device (str): Device to run the model on ('cpu' or 'cuda').
            iou_thres (float): IOU threshold for non-maximum suppression.
            conf_thres (float): Confidence threshold for detections.
        """
        self.device = device
        self.char_model, self.names = self.load_model(weights_path)
        self.size = size
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres

    def detect(self, frame, bb_scale=False):
        """
        Detect objects in a given image frame.

        Args:
            frame (np.ndarray): The input image in BGR format.
            bb_scale (bool): Whether to scale bounding boxes to the original image size.

        Returns:
            tuple: A tuple containing:
                - results (list): List of detected objects with class name, confidence, and bounding box.
                - resized_img (np.ndarray): The resized image used for detection.
        """
        results, resized_img = self.char_detection_yolo(frame, bb_scale=bb_scale)
        return results, resized_img

    def preprocess_image(self, original_image):
        """
        Preprocess the image before feeding it into the YOLO model.

        Args:
            original_image (np.ndarray): The original BGR image.

        Returns:
            tuple: A tuple containing:
                - image (torch.Tensor): Preprocessed image tensor.
                - resized_img (np.ndarray): The resized image used for input.
        """
        resized_img = self.ResizeImg(original_image, size=self.size)
        image = resized_img.copy()[:, :, ::-1].transpose(2, 0, 1)  # Convert BGR to RGB, and HWC to CHW
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).to(self.device).float() / 255.0
        if image.ndimension() == 3:
            image = image.unsqueeze(0)
        return image, resized_img

    def char_detection_yolo(self, image, classes=None, agnostic_nms=True, max_det=1000, bb_scale=False):
        """
        Perform YOLO-based object detection on an image.

        Args:
            image (np.ndarray): Input BGR image.
            classes (list or None): Class indices to filter detections. None means all classes.
            agnostic_nms (bool): Whether NMS should be class-agnostic.
            max_det (int): Maximum number of detections per image.
            bb_scale (bool): Whether to scale bounding boxes back to the original image size.

        Returns:
            tuple: A tuple containing:
                - results (list): List of detections [class_name, confidence, (x1, y1, x2, y2)].
                - resized_img (np.ndarray): The resized image used for detection.
        """
        img, resized_img = self.preprocess_image(image.copy())
        pred = self.char_model(img, augment=False)[0]
        detections = non_max_suppression(pred,
                                          conf_thres=self.conf_thres,
                                          iou_thres=self.iou_thres,
                                          classes=classes,
                                          agnostic=agnostic_nms,
                                          multi_label=True,
                                          labels=(),
                                          max_det=max_det)

        results = []
        for i, det in enumerate(detections):
            if bb_scale:
                det[:, :4] = scale_coords(resized_img.shape, det[:, :4], image.shape).round()
            det = det.tolist()
            if len(det):
                for *xyxy, conf, cls in det:
                    result = [self.names[int(cls)], str(conf), (xyxy[0], xyxy[1], xyxy[2], xyxy[3])]
                    results.append(result)
        return results, resized_img

    def ResizeImg(self, img, size):
        """
        Resize an image while keeping aspect ratio and padding with black pixels if necessary.

        Args:
            img (np.ndarray): Input BGR image.
            size (tuple): Target size (height, width).

        Returns:
            np.ndarray: Resized and padded image.
        """
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
        """
        Load a YOLOv5 model from a .pt weights file.

        Args:
            path (str): Path to the model weights file.
            train (bool): Whether to load the model in training mode.

        Returns:
            tuple: A tuple containing:
                - model (torch.nn.Module): The loaded model.
                - names (list): List of class names in the model.
        """
        model = attempt_load(path, map_location=self.device)
        names = model.module.names if hasattr(model, 'module') else model.names
        model.train() if train else model.eval()
        return model, names
