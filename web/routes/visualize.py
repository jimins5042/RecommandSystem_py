"""
POST /visualize/ — YOLO 탐지. 백본과 무관한 공용 시각화 엔드포인트.
"""
from __future__ import annotations

import logging
from io import BytesIO

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from web.detection import detect_objects, yolo_model

router = APIRouter(tags=["visualize"])
logger = logging.getLogger(__name__)


@router.post("/visualize/")
async def visualize(
    file: UploadFile = File(...),
):
    logger.info(f"[visualize] 요청 수신: {file.filename}")
    image = Image.open(BytesIO(await file.read())).convert("RGB")

    if yolo_model is None:
        logger.warning("[visualize] YOLO 미로드 — 감지 결과 비어있음")
        detections = []
    else:
        detections = detect_objects(image)

    logger.info(f"[visualize] 감지: {len(detections)}개")
    return JSONResponse(content={"detections": detections})
