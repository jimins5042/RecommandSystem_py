"""
백본별 특징점 추출 엔드포인트.

  GET  /backbones                              — 등록된 백본 목록 + 로드 상태
  POST /backbones/{name}/process-image/        — YOLO crop + 백본 특징점 추출
  POST /backbones/{name}/process-image/crop/   — 크롭 스킵 (이미 크롭된 입력)
"""
from __future__ import annotations

import base64
import logging
from io import BytesIO

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from api import REGISTRY, Backbone
from web.detection import detect_and_crop

router = APIRouter(tags=["backbones"])
logger = logging.getLogger(__name__)


def _get_backbone_or_404(name: str) -> Backbone:
    backbone = REGISTRY.get(name)
    if backbone is None:
        raise HTTPException(404, f"Unknown backbone: '{name}'. Available: {list(REGISTRY.keys())}")
    if not backbone.is_loaded():
        raise HTTPException(503, f"Backbone '{name}' model not loaded")
    return backbone


@router.get("/backbones")
async def list_backbones():
    """등록된 백본 목록과 로드 상태."""
    return [
        {
            "name": b.name,
            "displayName": b.display_name,
            "loaded": b.is_loaded(),
            "inputSize": list(b.input_size),
        }
        for b in REGISTRY.values()
    ]


@router.post("/backbones/{name}/process-image/")
async def process_image(
    name: str,
    file: UploadFile = File(...),
    use_rembg: bool = False,
    rembg_model: str = "u2net",
    alpha_matting: bool = False,
):
    backbone = _get_backbone_or_404(name)
    logger.info(f"[{name}/process-image] 요청 수신: {file.filename}")

    image = Image.open(BytesIO(await file.read())).convert("RGB")
    cropped, det_class, conf, coord, all_dets = detect_and_crop(
        image, use_rembg=use_rembg, rembg_model=rembg_model, alpha_matting=alpha_matting,
    )

    output = backbone.extract(cropped)

    return JSONResponse(content={
        "order": output.order,
        "featuresBase64": base64.b64encode(output.features_bytes).decode("utf-8"),
        "detectedClass": det_class,
        "confidence": round(conf, 4) if conf is not None else None,
        "coordinate": coord,
        "detections": all_dets,
    })


@router.post("/backbones/{name}/process-image/crop/")
async def process_image_crop(name: str, file: UploadFile = File(...)):
    backbone = _get_backbone_or_404(name)
    logger.info(f"[{name}/process-image/crop] 요청 수신: {file.filename}")

    image = Image.open(BytesIO(await file.read())).convert("RGB")
    output = backbone.extract(image)

    return JSONResponse(content={
        "order": output.order,
        "featuresBase64": base64.b64encode(output.features_bytes).decode("utf-8"),
    })
