"""Rice grain counting via classical CV.

Pipeline (why each step, not just what it does):

1. Grayscale + Gaussian blur -> suppress sensor noise before thresholding.
2. Otsu threshold, tightened by a brightness-percentile floor -> rice on a
   contrasting tray/paper background separates cleanly on brightness alone
   in principle, but a real phone photo rarely has a uniform background:
   glare, a light gradient, or a scratch/smudge on the surface can be just
   bright enough that plain Otsu lumps them in with the grains as one
   "foreground" class. Real grains under typical lighting (especially with
   flash) are near-saturated highlights, dramatically brighter than that
   kind of surface noise -- so the threshold is pulled toward whichever
   brightness extreme grains sit at (a percentile of the whole image),
   using Otsu's own value only as a starting point, never a looser one.
3. Morphological opening + a per-blob solidity check -> strips tiny speckle
   noise and rejects ragged, non-convex fragments (bits of a scratch or
   shadow edge that survive thresholding) that a real grain's smooth
   silhouette would never produce.
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
FOREGROUND_PERCENTILE = 92          # assume grains cover at most ~8% of the frame; tightens Otsu
MIN_GRAIN_SOLIDITY = 0.65           # contour area / convex-hull area; rejects ragged non-grain fragments
MIN_GRAIN_ABSOLUTE_AREA_PX = 40     # floor under the ratio-based minimum: at only a few px, sensor
                                     # noise/JPEG artifacts are trivially "solid" too small to shape-filter


@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int
    confidence: float


def _strict_foreground_mask(blurred: np.ndarray) -> np.ndarray:
    """Threshold the image into grain=255/background=0, tightened against
    uneven real-world lighting.

    Otsu alone finds *a* split point in the brightness histogram, but a real
    photo's "background" isn't one clean class -- it can span dim table to a
    lit-up scratch or glare, all dimmer than an actual grain highlight.
    Otsu can end up splitting between background and (scratch+grains)
    instead of between (background+scratch) and grains.

    Fix: figure out which side is the minority (grain) class the same way
    Otsu would, then tighten the cut toward that class's own brightness
    extreme -- using a percentile of the whole image as a floor/ceiling that
    Otsu's value is only allowed to move toward, never away from. Percentile
    is based on FOREGROUND_PERCENTILE assuming grains are a small minority
    of the frame, which holds for any "grains scattered on a background"
    photo without needing scene-specific tuning.
    """
    otsu_value, otsu_binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = np.count_nonzero(otsu_binary) / otsu_binary.size

    if white_ratio > 0.5:
        # Grains are the dark class -- tighten downward toward the dark extreme.
        floor = np.percentile(blurred, 100 - FOREGROUND_PERCENTILE)
        strict_value = min(otsu_value, floor)
        _, binary = cv2.threshold(blurred, strict_value, 255, cv2.THRESH_BINARY_INV)
    else:
        # Grains are the bright class -- tighten upward toward the bright extreme.
        ceiling = np.percentile(blurred, FOREGROUND_PERCENTILE)
        strict_value = max(otsu_value, ceiling)
        _, binary = cv2.threshold(blurred, strict_value, 255, cv2.THRESH_BINARY)

    return binary


def _is_grain_shaped(contour) -> bool:
    """A real grain's silhouette is a smooth, solid blob. A fragment of a
    scratch, shadow edge, or reflection that survives thresholding tends to
    be ragged/non-convex instead -- solidity (contour area / convex-hull
    area) tells them apart cheaply.
    """
    area = cv2.contourArea(contour)
    if area <= 0:
        return False
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    if hull_area == 0:
        return False
    return (area / hull_area) >= MIN_GRAIN_SOLIDITY


def _box_from_mask(mask: np.ndarray, offset_x: int = 0, offset_y: int = 0) -> "BoundingBox | None":
    """Finds the mask's contour, rejects non-grain-shaped ones, and returns
    a BoundingBox in full-image coordinates -- or None if it fails the
    shape check.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if not _is_grain_shaped(contour):
        return None
    x, y, w, h = cv2.boundingRect(contour)
    return _box_from_component(offset_x + x, offset_y + y, w, h)


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
    min_area = max(image_area * MIN_GRAIN_AREA_RATIO, MIN_GRAIN_ABSOLUTE_AREA_PX)
    max_area = image_area * MAX_GRAIN_AREA_RATIO

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Sanity gate: checked on the LOOSE (untightened) Otsu split, deliberately
    # separate from the strict mask below. A real "grains scattered on a
    # plain background" photo splits nowhere near 50/50 even under plain
    # Otsu -- grains are a small minority of the frame. A texture-less or
    # noisy image (blank wall, sensor noise, bad lighting) has no true
    # bimodal separation, so plain Otsu lands close to a 50/50 split. Using
    # the loose split for this check (rather than the strict mask, which is
    # tightened for a different purpose -- see _strict_foreground_mask)
    # keeps the two concerns independent: tightening the detection threshold
    # to reject a bright scratch shouldn't also blind this gate to noise.
    _, loose_binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    loose_ratio = np.count_nonzero(loose_binary) / loose_binary.size
    minority_ratio = min(loose_ratio, 1 - loose_ratio)
    if minority_ratio > MAX_FOREGROUND_RATIO:
        return []

    binary = _strict_foreground_mask(blurred)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPENING_KERNEL_SIZE, OPENING_KERNEL_SIZE))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

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
            # Already grain-sized on its own -- report directly if it's
            # actually grain-shaped (rejects compact-but-ragged noise blobs).
            component_mask = np.uint8(labels == label) * 255
            box = _box_from_mask(component_mask)
            if box is not None:
                boxes.append(box)
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
            box = _box_from_mask(sub_mask, offset_x=x1, offset_y=y1)
            if box is not None:
                boxes.append(box)

    return boxes


def _box_from_component(x: int, y: int, w: int, h: int) -> BoundingBox:
    aspect = max(w, h) / max(1, min(w, h))
    # Confidence heuristic: how "grain-shaped" (elongated) the box is, scaled
    # into 0..1. Not a learned probability -- a quality signal for the UI.
    confidence = 1.0 if 1.5 <= aspect <= 5.0 else 0.6
    return BoundingBox(x=int(x), y=int(y), width=int(w), height=int(h), confidence=confidence)
