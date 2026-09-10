"""Rice grain counting via classical CV.

Pipeline (why each step, not just what it does):

1. Grayscale + Gaussian blur -> suppress sensor noise before thresholding.
2. Otsu threshold -> rice on a contrasting tray/paper background separates
   cleanly on brightness alone; Otsu picks the split point automatically so
   this works across different lighting without a magic constant.
3. Morphological opening -> strips tiny speckle noise (dust, sensor grain)
   that would otherwise be counted as grains.
4. Per-blob distance transform + watershed -> the actual hard part. Rice
   grains touch and overlap in almost every real photo, so a plain "count
   contours" pass undercounts by merging touching grains into one blob.
   Distance-transform peaks mark each grain's widest point; flooding from
   those peaks with watershed splits touching grains at their narrowest
   connection.

   Crucially, this is done PER CONNECTED COMPONENT, not once globally. A
   single relative threshold over the whole image is fragile the moment the
   frame contains anything larger than a grain (a stray large object, a
   shadow blob, a ruler) -- its one big distance-transform peak swamps the
   relative threshold and every real grain peak below it disappears. Working
   blob-by-blob means each grain cluster is judged against its own scale.
5. Small isolated blobs (already grain-sized) skip watershed entirely --
   splitting a lone grain by its own internal curvature would over-count.
   Oversized blobs that don't split into multiple internal peaks are
   discarded as "not rice" rather than reported as one giant grain.
"""
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

# Tunable knobs. Defaults assume a phone photo of rice on a plain, contrasting
# tray taken from ~20-30cm. They scale relative to the image's own resolution
# rather than being hardcoded pixel counts, so they hold across camera specs.
MIN_GRAIN_AREA_RATIO = 0.00002    # a grain must be at least this fraction of image area
MAX_GRAIN_AREA_RATIO = 0.006      # ...and at most this fraction (rejects merged/non-rice blobs)
OPENING_KERNEL_SIZE = 3
DIST_TRANSFORM_THRESH_RATIO = 0.5  # fraction of a blob's own max distance-transform value
CLUSTER_PADDING = 3                # px of padding around each blob's crop for watershed
MAX_FOREGROUND_RATIO = 0.35        # above this, the scene isn't "sparse objects on a background"


@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int
    confidence: float


def _order_background_foreground(binary: np.ndarray) -> np.ndarray:
    """Otsu can invert which class (0/255) is foreground depending on the
    image. Rice grains are assumed to be the minority-area class; flip the
    mask if that assumption doesn't hold so foreground is always 255=grain.
    """
    white_ratio = np.count_nonzero(binary) / binary.size
    return cv2.bitwise_not(binary) if white_ratio > 0.5 else binary


def _split_cluster(mask_crop: np.ndarray) -> List[np.ndarray]:
    """Given a binary mask crop containing one or more touching grains,
    return a list of per-grain binary masks (same shape as mask_crop) using
    a local distance-transform + watershed split. Returns [] if the blob
    doesn't decompose into multiple peaks (i.e. it isn't a grain cluster).
    """
    dist = cv2.distanceTransform(mask_crop, cv2.DIST_L2, 5)
    max_dist = dist.max()
    if max_dist == 0:
        return []

    _, sure_fg = cv2.threshold(dist, DIST_TRANSFORM_THRESH_RATIO * max_dist, 255, 0)
    sure_fg = np.uint8(sure_fg)

    num_peaks, peak_markers = cv2.connectedComponents(sure_fg)
    if num_peaks <= 2:
        # <=1 real peak (0 = background) -> can't split; not a multi-grain cluster.
        return []

    sure_bg = cv2.dilate(mask_crop, np.ones((3, 3), np.uint8), iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)

    markers = peak_markers + 1
    markers[unknown == 255] = 0

    color_crop = cv2.cvtColor(mask_crop, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color_crop, markers)

    grain_masks = []
    for label in range(2, markers.max() + 1):
        sub_mask = np.uint8(markers == label) * 255
        if cv2.countNonZero(sub_mask) > 0:
            grain_masks.append(sub_mask)
    return grain_masks


def detect_rice(image_bgr: np.ndarray) -> List[BoundingBox]:
    height, width = image_bgr.shape[:2]
    image_area = height * width
    min_area = image_area * MIN_GRAIN_AREA_RATIO
    max_area = image_area * MAX_GRAIN_AREA_RATIO

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = _order_background_foreground(binary)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPENING_KERNEL_SIZE, OPENING_KERNEL_SIZE))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Sanity gate: a real "grains scattered on a plain tray" photo has a
    # clearly sparse foreground. A texture-less or noisy photo (blank wall,
    # sensor/JPEG noise, bad lighting) still gives Otsu *some* split point,
    # which without this check would get reported as a field of tiny
    # "grains" instead of correctly falling through to no_detections.
    foreground_ratio = np.count_nonzero(opened) / opened.size
    if foreground_ratio > MAX_FOREGROUND_RATIO:
        return []

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)

    boxes: List[BoundingBox] = []

    for label in range(1, num_labels):  # 0 = background
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue  # noise speckle

        x0 = stats[label, cv2.CC_STAT_LEFT]
        y0 = stats[label, cv2.CC_STAT_TOP]
        w0 = stats[label, cv2.CC_STAT_WIDTH]
        h0 = stats[label, cv2.CC_STAT_HEIGHT]

        if area <= max_area:
            # Already grain-sized on its own -- report directly, no split needed.
            boxes.append(_box_from_component(x0, y0, w0, h0))
            continue

        # Larger than one grain: could be several touching grains, or a
        # non-rice object entirely. Crop with padding and try to split it.
        pad = CLUSTER_PADDING
        x1, y1 = max(0, x0 - pad), max(0, y0 - pad)
        x2, y2 = min(width, x0 + w0 + pad), min(height, y0 + h0 + pad)
        component_mask = np.uint8(labels[y1:y2, x1:x2] == label) * 255

        grain_masks = _split_cluster(component_mask)
        if not grain_masks:
            continue  # a solid oversized blob (e.g. a stray object) -- not rice

        for sub_mask in grain_masks:
            sub_area = cv2.countNonZero(sub_mask)
            if sub_area < min_area or sub_area > max_area:
                continue
            contours, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            sx, sy, sw, sh = cv2.boundingRect(max(contours, key=cv2.contourArea))
            boxes.append(_box_from_component(x1 + sx, y1 + sy, sw, sh))

    return boxes


def _box_from_component(x: int, y: int, w: int, h: int) -> BoundingBox:
    aspect = max(w, h) / max(1, min(w, h))
    # Confidence heuristic: how "grain-shaped" (elongated) the box is, scaled
    # into 0..1. Not a learned probability -- a quality signal for the UI.
    confidence = 1.0 if 1.5 <= aspect <= 5.0 else 0.6
    return BoundingBox(x=int(x), y=int(y), width=int(w), height=int(h), confidence=confidence)
