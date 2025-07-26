import cv2
import numpy as np


def postprocess_mask(mask, apply_gaussian=True, kernel_size=5):
    """
    mask: binary uint8 mask (0 or 255)
    """
    if apply_gaussian:
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def hough_lines(mask, canny_thresh1=50, canny_thresh2=150):
    edges = cv2.Canny(mask, canny_thresh1, canny_thresh2)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=10,
    )
    line_img = np.zeros_like(mask)
    angles = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(line_img, (x1, y1), (x2, y2), 255, 2)
            angles.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    return line_img, float(np.median(angles)) if angles else 0.0


def process_prediction(pred_tensor):
    """
    pred_tensor: torch tensor [1,1,H,W], sigmoid output
    returns: cleaned mask, line image, median angle
    """
    mask = pred_tensor.squeeze().cpu().numpy()
    binary = (mask > 0.5).astype(np.uint8) * 255
    clean = postprocess_mask(binary)
    lines, angle = hough_lines(clean)
    return clean, lines, angle
