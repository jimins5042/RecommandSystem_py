"""
POST /visualize/ — YOLO 탐지 + (옵션) rembg. 백본과 무관한 공용 시각화 엔드포인트.
"""
from __future__ import annotations

import base64
import logging
from io import BytesIO

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from web.detection import apply_rembg, detect_objects, yolo_model

router = APIRouter(tags=["visualize"])
logger = logging.getLogger(__name__)


@router.post("/visualize/")
async def visualize(
    file: UploadFile = File(...),
    use_rembg: bool = False,
    model: str = "u2net",
    alpha_matting: bool = False,
):
    logger.info(f"[visualize] 요청 수신: {file.filename} (rembg={use_rembg})")
    image = Image.open(BytesIO(await file.read())).convert("RGB")

    rembg_base64 = None
    target = image
    if use_rembg:
        image_rgba, target = apply_rembg(image, model=model, alpha_matting=alpha_matting)
        buf = BytesIO()
        image_rgba.save(buf, format="PNG")
        rembg_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    if yolo_model is None:
        logger.warning("[visualize] YOLO 미로드 — 감지 결과 비어있음")
        detections = []
    else:
        detections = detect_objects(target)

    logger.info(f"[visualize] 감지: {len(detections)}개")
    return JSONResponse(content={"detections": detections, "rembgImage": rembg_base64})
