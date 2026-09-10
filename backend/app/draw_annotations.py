"""Renders numbered markers onto a copy of the source image. Used by the
optional /detect/annotated debug endpoint; the mobile app normally draws its
own overlay from the JSON detections (see mobile/src/components/DetectionOverlay.tsx)
so it can stay crisp/scalable on any screen size instead of depending on a
server-rasterized image.
"""
from typing import List

import cv2
import numpy as np

from .schemas import Detection

RICE_COLOR = (60, 200, 60)   # BGR green
CONE_COLOR = (30, 90, 220)   # BGR orange


def draw_annotations(image_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray:
    annotated = image_bgr.copy()
    line_thickness = max(2, annotated.shape[1] // 500)
    font_scale = max(0.5, annotated.shape[1] / 1600)

    for det in detections:
        color = RICE_COLOR if det.kind == "rice" else CONE_COLOR
        top_left = (det.x, det.y)
        bottom_right = (det.x + det.width, det.y + det.height)
        cv2.rectangle(annotated, top_left, bottom_right, color, line_thickness)

        label = str(det.id)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        label_origin = (det.x, max(0, det.y - 4))
        cv2.rectangle(
            annotated,
            (label_origin[0], label_origin[1] - text_h - 4),
            (label_origin[0] + text_w + 6, label_origin[1]),
            color,
            -1,
        )
        cv2.putText(
            annotated, label, (label_origin[0] + 3, label_origin[1] - 3),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA,
        )

    return annotated
