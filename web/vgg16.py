import base64
import json
import os
from io import BytesIO

import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from rembg import remove

from shared import (
    MODEL_DIR, logger,
    YOLO_CONF_THRESHOLD, CLASS_NAMES, yolo_model,
    get_rembg_session, detect_and_crop,
)

router = APIRouter(prefix="/vgg16")

# ── VGG16 ONNX 모델 로드 ──
session_full   = ort.InferenceSession(os.path.join(MODEL_DIR, "vgg16_full.onnx"))
session_block5 = ort.InferenceSession(os.path.join(MODEL_DIR, "vgg16_block5_conv3.onnx"))
logger.info(f"VGG16 ONNX models loaded from {MODEL_DIR}")


def _preprocess(img: np.ndarray) -> np.ndarray:
    """RGB→BGR 변환 후 ImageNet 채널별 평균 차감 (TF caffe 모드)"""
    img = img[..., ::-1].copy()
    img[..., 0] -= 103.939
    img[..., 1] -= 116.779
    img[..., 2] -= 123.68
    return img


def _load_and_preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((224, 224))
    img = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    return _preprocess(img)


def _extract_features(img: np.ndarray) -> bytes:
    input_name = session_full.get_inputs()[0].name
    features = session_full.run(None, {input_name: img})[0].flatten()
    nonzero = features[features != 0]
    mean_val = np.mean(nonzero) if len(nonzero) > 0 else 0
    binary = np.where(features >= mean_val, 1, 0)
    return np.packbits(binary).tobytes()


def _extract_order(img: np.ndarray) -> str:
    input_name = session_block5.get_inputs()[0].name
    fmaps = session_block5.run(None, {input_name: img})[0].squeeze()  # (14,14,512)
    means = {f"{i}": np.mean(fmaps[:, :, i]) for i in range(512)}
    top25 = [int(k) for k, _ in sorted(means.items(), key=lambda x: x[1], reverse=True)[:25]]
    return json.dumps(top25)


# ── 엔드포인트 ──

@router.post("/visualize/")
async def visualize_image(
    file: UploadFile = File(...),
    use_rembg: bool = True,
    model: str = "u2net",
    alpha_matting: bool = False,
):
    logger.info(f"[vgg16/visualize] 요청 수신: {file.filename}")
    image = Image.open(BytesIO(await file.read())).convert("RGB")

    rembg_base64 = None
    processed_image = image

    if use_rembg:
        session = get_rembg_session(model)
        image_rgba = remove(image, session=session, alpha_matting=alpha_matting)
        buffered = BytesIO()
        image_rgba.save(buffered, format="PNG")
        rembg_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        processed_image = Image.new("RGB", image_rgba.size, (255, 255, 255))
        processed_image.paste(image_rgba, mask=image_rgba.split()[3])

    results = yolo_model.predict(np.array(processed_image), conf=YOLO_CONF_THRESHOLD, verbose=False)
    boxes = results[0].boxes

    detections = []
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            b = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0].item())
            detections.append({
                "class": CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else "unknown",
                "confidence": round(box.conf[0].item(), 4),
                "coordinate": [int(b[0]), int(b[1]), int(b[2]), int(b[3])]
            })

    logger.info(f"[vgg16/visualize] 감지: {len(detections)}개")
    return JSONResponse(content={"detections": detections, "rembgImage": rembg_base64})


@router.post("/process-image/")
async def process_image(
    file: UploadFile = File(...),
    use_rembg: bool = False,
    model: str = "u2net",
    alpha_matting: bool = False,
):
    logger.info(f"[vgg16/process-image] 요청 수신: {file.filename}")
    image = Image.open(BytesIO(await file.read())).convert("RGB")

    cropped, detectedClass, confidence, coordinate, all_detections = detect_and_crop(
        image, use_rembg=use_rembg, model=model, alpha_matting=alpha_matting
    )
    logger.info(f"[vgg16/process-image] YOLO: class={detectedClass}, conf={confidence}")

    img = _load_and_preprocess(cropped)
    features_base64 = base64.b64encode(_extract_features(img)).decode("utf-8")
    order = _extract_order(img)

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
    image = Image.open(BytesIO(await file.read())).convert("RGB")

    img = _load_and_preprocess(image)
    features_base64 = base64.b64encode(_extract_features(img)).decode("utf-8")
    order = _extract_order(img)

    return JSONResponse(content={"order": order, "featuresBase64": features_base64})
