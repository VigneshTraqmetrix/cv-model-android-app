"""Blur detection so we can reject unusable photos before running CV on them.

Uses the variance of the Laplacian: a sharp image has lots of high-frequency
edge content (high variance); a blurry one is smoothed out (low variance).
This is the standard, cheap heuristic for this problem and is good enough to
gate a mobile-captured photo before spending time on contour/model inference.
"""
import cv2
import numpy as np

# Empirically, phone photos of small textured objects (rice/cones) score
# well above this on a focused shot and collapse below it when out of focus.
DEFAULT_BLUR_THRESHOLD = 60.0


def compute_sharpness(gray_image: np.ndarray) -> float:
    return cv2.Laplacian(gray_image, cv2.CV_64F).var()


def is_blurry(image_bgr: np.ndarray, threshold: float = DEFAULT_BLUR_THRESHOLD) -> tuple[bool, float]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    score = compute_sharpness(gray)
    return score < threshold, score
