import base64
import json
import os
from io import BytesIO

import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from shared import MODEL_DIR, logger, detect_and_crop

router = APIRouter(prefix="/efficientNet")

# ── EfficientNet-B0 ONNX 모델 로드 ──
_EFFICIENTNET_PATH   = os.path.join(MODEL_DIR, "efficientnet-b0-feat.onnx")
_STAGE1_NODE = "/features/features.7/features.7.0/block/block.3/block.3.0/Conv_output_0"
_STAGE2_NODE = "/Flatten_output_0"

if os.path.exists(_EFFICIENTNET_PATH):
    _session = ort.InferenceSession(_EFFICIENTNET_PATH)
    logger.info(f"EfficientNet-B0 model loaded from {_EFFICIENTNET_PATH}")
else:
    _session = None
    logger.warning(f"EfficientNet-B0 model not found at {_EFFICIENTNET_PATH}. Run build_feature_model.py first.")

# ── 전처리 상수 (ImageNet, NCHW) ──
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _preprocess(image: Image.Image) -> np.ndarray:
    """PIL Image → NCHW float32 [1,3,224,224], ImageNet 정규화"""
    img = np.array(image.resize((224, 224)), dtype=np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims((img - _MEAN) / _STD, axis=0).astype(np.float32)


def _extract_features(image: Image.Image):
    img_array = _preprocess(image)
    input_name = _session.get_inputs()[0].name
    stage1_map, embedding = _session.run([_STAGE1_NODE, _STAGE2_NODE], {input_name: img_array})

    # Stage 1: (1,320,7,7) → GAP → Top-25
    channel_scores = stage1_map[0].mean(axis=(1, 2))
    order = json.dumps(np.argsort(channel_scores)[::-1][:25].tolist())

    # Stage 2: 1280D → 평균 이진화 → 160 byte
    emb = embedding.flatten()
    mean_val = np.mean(emb[emb != 0]) if np.any(emb != 0) else 0.0
    features_bytes = np.packbits(np.where(emb >= mean_val, 1, 0)).tobytes()
    features_base64 = base64.b64encode(features_bytes).decode("utf-8")

    return order, features_base64
# ── 엔드포인트 ──

@router.post("/process-image/")
async def process_image(file: UploadFile = File(...), use_yolo: bool = True):
    """
    Stage 1 (order)         : 7x7x320 → GAP → Top-25 인덱스
    Stage 2 (featuresBase64): 1280D   → 평균 이진화 → 160 byte
    """
    if _session is None:
        return JSONResponse(status_code=503, content={"error": "EfficientNet model not loaded. Run build_feature_model.py first."})

    logger.info(f"[efficientNet/process-image] 요청 수신: {file.filename}, use_yolo={use_yolo}")
    image = Image.open(BytesIO(await file.read())).convert("RGB")

    detectedClass, confidence, coordinate, all_detections = None, None, None, []
    if use_yolo:
        image, detectedClass, confidence, coordinate, all_detections = detect_and_crop(image)
    logger.info(f"[efficientNet/process-image] YOLO: class={detectedClass}, conf={confidence}")

    img_array = _preprocess(image)
    input_name = _session.get_inputs()[0].name
    stage1_map, embedding = _session.run([_STAGE1_NODE, _STAGE2_NODE], {input_name: img_array})

    # Stage 1: (1,320,7,7) → GAP → Top-25
    channel_scores = stage1_map[0].mean(axis=(1, 2))
    order = json.dumps(np.argsort(channel_scores)[::-1][:25].tolist())

    # Stage 2: 1280D → 평균 이진화 → 160 byte
    emb = embedding.flatten()
    mean_val = np.mean(emb[emb != 0]) if np.any(emb != 0) else 0.0
    features_bytes = np.packbits(np.where(emb >= mean_val, 1, 0)).tobytes()
    features_base64 = base64.b64encode(features_bytes).decode("utf-8")

    logger.info(f"[efficientNet/process-image] 완료: features={len(features_bytes)}byte")
    return JSONResponse(content={
        "order": order,
        "featuresBase64": features_base64,
        "detectedClass": detectedClass,
        "confidence": round(confidence, 4) if confidence is not None else None,
        "coordinate": coordinate,
        "detections": all_detections,
    })

@router.post("/process-image/crop/")
async def process_image_crop(file: UploadFile = File(...)):
    logger.info(f"[vgg16/process-image/crop] 요청 수신: {file.filename}")
    order, features_base64 = _extract_features(file)

    return JSONResponse(content={"order": order, "featuresBase64": features_base64})