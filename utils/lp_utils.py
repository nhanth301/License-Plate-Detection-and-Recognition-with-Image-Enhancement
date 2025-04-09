import cv2
import numpy as np


def sort_license_plate_detections(yolo_output):
    """
    Sorts YOLO license plate detections in reading order (left to right, top to bottom).

    This function takes the output from a YOLO object detector and sorts the 
    detected characters based on their position to match the typical reading 
    order of license plates.

    Args:
        yolo_output (List[List[Union[str, float, Tuple[int, int, int, int]]]]): 
            A list of detections. Each detection is a list containing:
            - class (str): Class label (e.g., character).
            - confidence (float): Confidence score of detection.
            - bbox (tuple): Bounding box in the form (x1, y1, x2, y2), where
              (x1, y1) is the top-left and (x2, y2) is the bottom-right corner.

    Returns:
        List[List[Union[str, float, Tuple[int, int, int, int]]]]: 
            Sorted list of detections in reading order.

    Example:
        >>> sorted_dets = sort_license_plate_detections(yolo_output)
    """
    if not yolo_output:
        return []

    # Compute center coordinates of each bounding box
    detections_with_centers = []
    for detection in yolo_output:
        char_class, confidence, bbox = detection
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        detections_with_centers.append({
            'detection': detection,
            'center_x': center_x,
            'center_y': center_y
        })

    # Sort by vertical center to group rows
    detections_with_centers.sort(key=lambda x: x['center_y'])

    rows = []
    current_row = [detections_with_centers[0]]
    y_threshold = max(10, (detections_with_centers[-1]['center_y'] -
                           detections_with_centers[0]['center_y']) / 5)

    for i in range(1, len(detections_with_centers)):
        if abs(detections_with_centers[i]['center_y'] -
               detections_with_centers[i - 1]['center_y']) > y_threshold:
            # Start a new row
            rows.append(current_row)
            current_row = [detections_with_centers[i]]
        else:
            current_row.append(detections_with_centers[i])

    rows.append(current_row)  # Add the last row

    # Sort each row left-to-right
    for row in rows:
        row.sort(key=lambda x: x['center_x'])

    # Flatten and return only original detection info
    sorted_detections = [item['detection'] for row in rows for item in row]

    return sorted_detections


def straighten_license_plate(plate_img):
    """
    Automatically straightens a potentially tilted license plate image.

    This function detects lines or contours in the image to estimate the skew
    angle and rotates the image to make the license plate appear straight.

    Args:
        plate_img (np.ndarray): Input image of the license plate 
                                (grayscale or BGR).

    Returns:
        np.ndarray: Rotated image with the license plate aligned horizontally.

    Example:
        >>> img = cv2.imread("plate.jpg")
        >>> straight_img = straighten_license_plate(img)
    """
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img.copy()

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=30,
        maxLineGap=10
    )

    angle = 0
    if lines is not None and len(lines) > 0:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                continue  # Skip vertical lines
            angle_rad = np.arctan2(dy, dx)
            angle_deg = np.degrees(angle_rad)
            if -45 < angle_deg < 45:
                angles.append(angle_deg)
        if angles:
            angle = np.median(angles)
    else:
        # Fallback using contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            angle = rect[2]
            if angle < -45:
                angle = 90 + angle
            elif angle > 45:
                angle = angle - 90

    # Rotate the image
    (h, w) = plate_img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        plate_img,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


if __name__ == '__main__':
    img = cv2.imread('image.png')
    corrected_img = straighten_license_plate(img)
    cv2.imwrite('corrected.png', corrected_img)
