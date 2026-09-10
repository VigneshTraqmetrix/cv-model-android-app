"""Sanity tests for the CV pipelines and the FastAPI endpoint, using a
synthetic image with known counts (see generate_test_image.py) so tests
don't depend on a real photo being checked into the repo.
"""
import io
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.rice_detector import detect_rice
from app.cone_detector import detect_cones
from app.blur_check import is_blurry
from tests.generate_test_image import make_test_image


def test_rice_detector_finds_most_grains():
    img, rice_boxes, _ = make_test_image()
    detected = detect_rice(img)
    # Classical CV with intentionally-overlapping grains won't hit 100%;
    # assert it's in a sane, useful range instead of an exact match.
    assert len(detected) >= 0.6 * len(rice_boxes)
    assert len(detected) <= len(rice_boxes)


def test_cone_detector_finds_all_cones():
    img, _, cone_boxes = make_test_image()
    detected = detect_cones(img)
    assert len(detected) == len(cone_boxes)


def test_rice_detector_ignores_cones():
    img, _, _ = make_test_image(n_rice=0, n_cones=3)
    detected = detect_rice(img)
    assert len(detected) == 0


def test_blur_detection_flags_heavily_blurred_image():
    img, _, _ = make_test_image()
    blurred_img = cv2.GaussianBlur(img, (51, 51), 0)
    sharp_flag, _ = is_blurry(img)
    blurry_flag, _ = is_blurry(blurred_img)
    assert not sharp_flag
    assert blurry_flag


def test_detect_endpoint_returns_counts():
    from fastapi.testclient import TestClient
    from app.main import app

    img, rice_boxes, cone_boxes = make_test_image()
    ok, buf = cv2.imencode(".jpg", img)
    assert ok

    client = TestClient(app)
    response = client.post(
        "/detect",
        files={"image": ("sample.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")},
        data={"mode": "auto"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["cone_count"] == len(cone_boxes)
    assert body["rice_count"] > 0
    assert len(body["detections"]) == body["rice_count"] + body["cone_count"]
    # Labels are 1-based and human readable, e.g. "Rice #1", "Cone #2"
    labels = [d["label"] for d in body["detections"]]
    assert "Cone #1" in labels


def test_detect_endpoint_flags_no_detections_on_blank_image():
    from fastapi.testclient import TestClient
    from app.main import app

    rng = np.random.default_rng(0)  # deterministic: avoid a flaky test
    blank = np.full((400, 400, 3), 128, dtype=np.uint8)
    # add a little texture so it isn't flagged as blurry instead
    noise = (rng.random((400, 400, 3)) * 20).astype(np.uint8)
    blank = cv2.add(blank, noise)
    ok, buf = cv2.imencode(".jpg", blank)
    assert ok

    client = TestClient(app)
    response = client.post(
        "/detect",
        files={"image": ("blank.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")},
        data={"mode": "auto"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("no_detections", "blurry")
