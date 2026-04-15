"""
POST /compare/process-image/ — 여러 백본을 동일 이미지로 병렬 실행 + 지연시간 측정.

백본 간 품질/속도 비교를 단일 요청으로 수행하기 위한 핵심 엔드포인트.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import time
from io import BytesIO

from fastapi import APIRouter, File, Query, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from api import REGISTRY, Backbone
from web.detection import detect_and_crop

router = APIRouter(tags=["compare"])
logger = logging.getLogger(__name__)


async def _run_backbone_timed(backbone: Backbone, image: Image.Image) -> dict:
    """ONNX 추론은 CPU 바운드라 executor 로 병렬화."""
    loop = asyncio.get_running_loop()
    start = time.perf_counter()
    try:
        output = await loop.run_in_executor(None, backbone.extract, image)
    except Exception as e:
        logger.exception(f"[compare] {backbone.name} 추론 실패")
        return {"error": str(e)}
    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "order": output.order,
        "featuresBase64": base64.b64encode(output.features_bytes).decode("utf-8"),
        "latencyMs": round(latency_ms, 2),
    }


@router.post("/compare/process-image/")
async def compare_process_image(
    file: UploadFile = File(...),
    backbones: list[str] | None = Query(None, description="비교할 백본 이름 목록. 생략 시 로드된 모든 백본."),
    use_rembg: bool = False,
    rembg_model: str = "u2net",
    alpha_matting: bool = False,
):
    """동일 이미지에 대해 지정된 백본들을 모두 실행하고 결과 + latency 반환."""
    # 타겟 결정
    if backbones:
        targets = [n for n in backbones if n in REGISTRY]
        unknown = set(backbones) - set(targets)
        if unknown:
            logger.warning(f"[compare] 알 수 없는 백본 요청 (무시): {unknown}")
    else:
        targets = [n for n, b in REGISTRY.items() if b.is_loaded()]

    if not targets:
        return JSONResponse(status_code=503, content={"error": "No loaded backbones available"})

    logger.info(f"[compare] 요청 수신: {file.filename} → {targets}")

    image = Image.open(BytesIO(await file.read())).convert("RGB")

    # YOLO crop + rembg 는 한 번만 수행 (공유)
    cropped, det_class, conf, coord, all_dets = detect_and_crop(
        image, use_rembg=use_rembg, rembg_model=rembg_model, alpha_matting=alpha_matting,
    )

    # 백본들을 병렬 실행
    tasks = [_run_backbone_timed(REGISTRY[n], cropped) for n in targets]
    results = await asyncio.gather(*tasks)

    return JSONResponse(content={
        "shared": {
            "detectedClass": det_class,
            "confidence": round(conf, 4) if conf is not None else None,
            "coordinate": coord,
            "detections": all_dets,
        },
        "results": dict(zip(targets, results)),
    })
