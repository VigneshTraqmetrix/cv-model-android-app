"""Pydantic response models shared by the API."""
from typing import List, Literal
from pydantic import BaseModel


class Detection(BaseModel):
    id: int                       # 1-based index used for the on-image label ("Rice #1")
    label: str                    # "Rice #1" / "Cone #2"
    kind: Literal["rice", "cone"]
    x: int                        # bounding box, in ORIGINAL image pixel space
    y: int
    width: int
    height: int
    confidence: float             # 0..1 (heuristic score for classical CV, model score for YOLO)


class DetectionResponse(BaseModel):
    status: Literal["ok", "no_detections", "blurry", "error"]
    message: str
    image_width: int
    image_height: int
    rice_count: int
    cone_count: int
    detections: List[Detection]
