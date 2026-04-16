"""
YOLO 상품 감지 (서빙 전용).

상위 라우터들이 공통으로 사용. 백본에 독립적.
"""
from __future__ import annotations

import logging
import os

import numpy as np
from PIL import Image
from ultralytics import YOLO

from config import CLASS_NAMES, MODEL_DIR, YOLO_CONF_THRESHOLD

logger = logging.getLogger(__name__)

# ── YOLO 모델 로드 ──
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
if os.path.exists(YOLO_MODEL_PATH):
    yolo_model = YOLO(YOLO_MODEL_PATH)
    logger.info(f"YOLO model loaded from {YOLO_MODEL_PATH}")
else:
    yolo_model = None
    logger.warning(f"YOLO model not found at {YOLO_MODEL_PATH}. Running without object detection.")


def detect_objects(image: Image.Image) -> list[dict]:
    """YOLO 전체 감지 결과를 dict 리스트로 반환 (crop 없음)."""
    if yolo_model is None:
        return []

    results = yolo_model.predict(np.array(image), conf=YOLO_CONF_THRESHOLD, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    w, h = image.size
    detections = []
    for box in boxes:
        b = box.xyxy[0].cpu().numpy()
        cls = int(box.cls[0].item())
        detections.append({
            "class": CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else "unknown",
            "confidence": round(box.conf[0].item(), 4),
            "coordinate": [int(max(0, b[0])), int(max(0, b[1])),
                           int(min(w, b[2])), int(min(h, b[3]))],
        })
    return detections


def detect_and_crop(image: Image.Image):
    """
    YOLO 로 상품 감지 후 confidence 최고 박스만 crop.
    반환: (cropped_image, detected_class, confidence, coordinate, all_detections)

    YOLO 미로드나 감지 실패 시 원본 이미지 + None 들 반환.
    """
    if yolo_model is None:
        return image, None, None, None, []

    results = yolo_model.predict(np.array(image), conf=YOLO_CONF_THRESHOLD, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return image, None, None, None, []

    w, h = image.size
    all_detections = []
    for box in boxes:
        b = box.xyxy[0].cpu().numpy()
        cls = int(box.cls[0].item())
        all_detections.append({
            "class": CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else "unknown",
            "confidence": round(box.conf[0].item(), 4),
            "coordinate": [int(max(0, b[0])), int(max(0, b[1])),
                           int(min(w, b[2])), int(min(h, b[3]))],
        })

    best_idx  = boxes.conf.argmax().item()
    best_box  = boxes.xyxy[best_idx].cpu().numpy().astype(int)
    best_conf = boxes.conf[best_idx].item()
    best_cls  = int(boxes.cls[best_idx].item())
    class_name = CLASS_NAMES[best_cls] if best_cls < len(CLASS_NAMES) else "unknown"

    x1, y1 = max(0, best_box[0]), max(0, best_box[1])
    x2, y2 = min(w, best_box[2]), min(h, best_box[3])
    cropped = image.crop((x1, y1, x2, y2))

    if cropped.size[0] == 0 or cropped.size[1] == 0:
        return image, None, None, None, all_detections

    return cropped, class_name, best_conf, [int(x1), int(y1), int(x2), int(y2)], all_detections
