"""Generates a synthetic test image with known counts of rice-grain-like
ellipses and cone-like triangles on a plain background, so the detectors can
be verified without needing a real photo. Run standalone: `python
generate_test_image.py` writes sample.png next to this file.
"""
import random
import cv2
import numpy as np


def make_test_image(width=1200, height=900, n_rice=40, n_cones=3, seed=7):
    random.seed(seed)
    # Dark tray background (as recommended for real captures) so pale rice
    # grains contrast cleanly against it -- mirrors a real photo setup
    # rather than the low-contrast light-on-light case.
    img = np.full((height, width, 3), 40, dtype=np.uint8)

    rice_boxes = []
    attempts = 0
    while len(rice_boxes) < n_rice and attempts < n_rice * 40:
        attempts += 1
        cx = random.randint(60, width - 60)
        cy = random.randint(60, height - 200)  # keep bottom strip for cones
        rw = random.randint(10, 14)
        rh = random.randint(28, 38)
        angle = random.randint(0, 180)

        # Reject placements that overlap an existing grain's center by too
        # much so we get a mix of isolated and lightly-touching grains
        # (touching grains are the whole reason watershed splitting exists).
        too_close = any(
            (cx - ex) ** 2 + (cy - ey) ** 2 < (min(rw, ew) * 0.6) ** 2
            for ex, ey, ew, eh, _ in rice_boxes
        )
        if too_close:
            continue

        shade = 200 + random.randint(-15, 15)
        cv2.ellipse(img, (cx, cy), (rw, rh), angle, 0, 360, (shade, shade, shade - 10), thickness=-1)
        rice_boxes.append((cx, cy, rw, rh, angle))

    cone_boxes = []
    for i in range(n_cones):
        cx = 150 + i * 350
        cy = height - 110
        pts = np.array([[cx, cy - 100], [cx - 70, cy + 80], [cx + 70, cy + 80]], np.int32)
        cv2.fillPoly(img, [pts], (30, 90, 220))  # orange-ish BGR cone
        cone_boxes.append(cv2.boundingRect(pts))

    return img, rice_boxes, cone_boxes


if __name__ == "__main__":
    img, rice_boxes, cone_boxes = make_test_image()
    cv2.imwrite("sample.png", img)
    print(f"Wrote sample.png with {len(rice_boxes)} rice grains and {len(cone_boxes)} cones")
