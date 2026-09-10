"""FastAPI backend for the cone + rice counter app.

Endpoints:
  GET  /health              liveness check
  POST /detect              run the CV pipeline(s), return JSON detections
  POST /detect/annotated     same as above but returns a JPEG with numbered
                             boxes drawn on it (handy for debugging /
                             sharing; the mobile app draws its own overlay
                             from the JSON so it stays crisp on-device)

`mode` (form field, default "auto"):
  "rice" -> run only the rice pipeline
  "cone" -> run only the cone pipeline
  "auto" -> run both and return whatever each finds

A single mode field rather than always running both lets the mobile app's
capture screen (which asks the user to pick "Rice" or "Cones" before
shooting, since the two need very different framing/zoom) skip the unused
pipeline entirely and save latency.
"""
import io
from typing import List, Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .blur_check import is_blurry
from .cone_detector import detect_cones
from .rice_detector import detect_rice
from .draw_annotations import draw_annotations
from .schemas import Detection, DetectionResponse

app = FastAPI(title="Cone & Rice Counter API", version="1.0.0")

# Dev-friendly CORS: the Expo app talks to this server over LAN/tunnel during
# development. Lock this down to your real origin(s) before shipping.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _read_image(file: UploadFile) -> np.ndarray:
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def _run_pipelines(image_bgr: np.ndarray, mode: str) -> List[Detection]:
    detections: List[Detection] = []
    next_id_rice, next_id_cone = 1, 1

    if mode in ("rice", "auto"):
        for box in detect_rice(image_bgr):
            detections.append(Detection(
                id=next_id_rice, label=f"Rice #{next_id_rice}", kind="rice",
                x=box.x, y=box.y, width=box.width, height=box.height,
                confidence=box.confidence,
            ))
            next_id_rice += 1

    if mode in ("cone", "auto"):
        for box in detect_cones(image_bgr):
            detections.append(Detection(
                id=next_id_cone, label=f"Cone #{next_id_cone}", kind="cone",
                x=box.x, y=box.y, width=box.width, height=box.height,
                confidence=box.confidence,
            ))
            next_id_cone += 1

    return detections


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect", response_model=DetectionResponse)
async def detect(
    image: UploadFile = File(...),
    mode: Literal["rice", "cone", "auto"] = Form("auto"),
):
    image_bgr = await _read_image(image)
    if image_bgr is None:
        return DetectionResponse(
            status="error", message="Could not decode image. Please retake the photo.",
            image_width=0, image_height=0, rice_count=0, cone_count=0, detections=[],
        )

    blurry, sharpness = is_blurry(image_bgr)
    if blurry:
        return DetectionResponse(
            status="blurry",
            message=f"Image looks too blurry to analyze (sharpness={sharpness:.1f}). Hold steady and retake.",
            image_width=image_bgr.shape[1], image_height=image_bgr.shape[0],
            rice_count=0, cone_count=0, detections=[],
        )

    detections = _run_pipelines(image_bgr, mode)
    rice_count = sum(1 for d in detections if d.kind == "rice")
    cone_count = sum(1 for d in detections if d.kind == "cone")

    if not detections:
        return DetectionResponse(
            status="no_detections",
            message="No cones or rice grains were detected. Make sure objects are well-lit and in frame.",
            image_width=image_bgr.shape[1], image_height=image_bgr.shape[0],
            rice_count=0, cone_count=0, detections=[],
        )

    return DetectionResponse(
        status="ok",
        message=f"Detected {rice_count} rice grain(s) and {cone_count} cone(s).",
        image_width=image_bgr.shape[1], image_height=image_bgr.shape[0],
        rice_count=rice_count, cone_count=cone_count, detections=detections,
    )


@app.post("/detect/annotated")
async def detect_annotated(
    image: UploadFile = File(...),
    mode: Literal["rice", "cone", "auto"] = Form("auto"),
):
    image_bgr = await _read_image(image)
    detections = _run_pipelines(image_bgr, mode)
    annotated = draw_annotations(image_bgr, detections)
    success, buffer = cv2.imencode(".jpg", annotated)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
