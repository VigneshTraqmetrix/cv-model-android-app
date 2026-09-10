"""Cone detection: a YOLOv8-nano detector when trained weights are present,
with a classical-CV fallback so the endpoint still works out of the box.

Why two paths:

- Cones are a macro object with huge appearance variety (color, material,
  size) -- there is no COCO-pretrained class for them, so a real deployment
  needs a small custom-trained YOLOv8n model (see `training/README.md` for
  the fine-tuning recipe: ~150-300 labeled photos is typically enough for a
  single, consistently-colored cone type).
- Until that model exists (or for a quick demo), we fall back to classical
  CV: HSV color thresholding for the cone's known color range + contour
  shape checks (roughly triangular, convex, taller than wide). This is the
  same "recognize a controlled object in a controlled scene" approach used
  for the rice pipeline, just tuned for a solid conical silhouette instead
  of many small grains.

Both paths return the same BoundingBox shape so main.py doesn't care which
one ran.
"""
import os
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "cone_yolov8n.pt")
YOLO_CONFIDENCE_THRESHOLD = 0.4

# --- Classical-CV fallback tuning -------------------------------------------------
# Default HSV range targets a bright orange safety-cone color. Adjust these
# for your actual cone color (e.g. a paper/plastic cone in a different hue)
# by sampling a few pixels from a sample photo -- see README "Calibrating
# the cone color range".
HSV_LOWER = np.array([5, 120, 90])
HSV_UPPER = np.array([25, 255, 255])
MIN_CONE_AREA_RATIO = 0.001   # a cone must be at least this fraction of image area
MAX_CONE_AREA_RATIO = 0.35    # ...and at most this fraction (rejects background/whole-frame blobs)


@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int
    confidence: float


def _yolo_available() -> bool:
    return os.path.isfile(MODEL_PATH)


_yolo_model = None


def _get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO  # imported lazily: optional dependency
        _yolo_model = YOLO(MODEL_PATH)
    return _yolo_model


def _detect_cones_yolo(image_bgr: np.ndarray) -> List[BoundingBox]:
    model = _get_yolo_model()
    results = model.predict(image_bgr, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD)
    boxes: List[BoundingBox] = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            boxes.append(BoundingBox(
                x=int(x1), y=int(y1),
                width=int(x2 - x1), height=int(y2 - y1),
                confidence=conf,
            ))
    return boxes


def _is_cone_shaped(contour) -> bool:
    """Cheap shape gate: a cone's silhouette is convex and taller than it is
    wide (unlike a sprawled grain cluster or a flat shadow)."""
    x, y, w, h = cv2.boundingRect(contour)
    if h == 0:
        return False
    aspect = h / w
    if not (0.9 <= aspect <= 2.6):
        return False

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    contour_area = cv2.contourArea(contour)
    if hull_area == 0:
        return False
    solidity = contour_area / hull_area  # near-1 for a clean convex cone silhouette
    return solidity > 0.85


def _detect_cones_classical(image_bgr: np.ndarray) -> List[BoundingBox]:
    height, width = image_bgr.shape[:2]
    image_area = height * width
    min_area = image_area * MIN_CONE_AREA_RATIO
    max_area = image_area * MAX_CONE_AREA_RATIO

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[BoundingBox] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        if not _is_cone_shaped(contour):
            continue
        x, y, w, h = cv2.boundingRect(contour)
        # Confidence heuristic scaled by how much of the bounding box the
        # colored region actually fills (a clean cone silhouette fills most
        # of its own bbox; a noisy blob fills less).
        fill_ratio = area / (w * h)
        confidence = float(min(1.0, 0.5 + fill_ratio * 0.5))
        boxes.append(BoundingBox(x=int(x), y=int(y), width=int(w), height=int(h), confidence=confidence))

    return boxes


def detect_cones(image_bgr: np.ndarray) -> List[BoundingBox]:
    if _yolo_available():
        try:
            return _detect_cones_yolo(image_bgr)
        except Exception:
            # A broken/incompatible weights file shouldn't take the whole
            # endpoint down -- degrade to the classical detector instead.
            pass
    return _detect_cones_classical(image_bgr)
