import json
import os
import logging
import base64
from io import BytesIO

# 기본 로깅 설정 추가 (로그 출력 보장)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. VGG16 모델 설정 (TensorFlow 대신 PyTorch 사용)
vgg16 = models.vgg16(weights='IMAGENET1K_V1')
vgg16.eval()

# Keras의 block5_conv3(ReLU 포함)와 유사한 지점까지 자르기
# vgg16.features[28] is Conv5_3, [29] is ReLU
model_mid = nn.Sequential(*list(vgg16.features)[:30]).to(device)
base_extractor = vgg16.features.to(device)

# PyTorch 이미지 전처리 설정
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 2. YOLO 모델 설정
# 도커 설정에서 정한 절대 경로(/opt/vgg16/RecommandSystem_py)를 우선 참조
BASE_DIR = os.environ.get("APP_HOME", os.path.dirname(os.path.abspath(__file__)))
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "model", "best.pt")
YOLO_CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.5"))
CLASS_NAMES = ["bag", "sunglasses", "food_drink", "shoes", "clothing"]

# 디버깅을 위한 경로 출력
logger.info(f"YOLO Search Path: {YOLO_MODEL_PATH}")

if os.path.exists(YOLO_MODEL_PATH):
    yolo_model = YOLO(YOLO_MODEL_PATH).to(device)
    logger.info(f"YOLO model loaded on {device}")
else:
    # 하드코딩된 절대 경로를 마지막 수단으로 시도
    fallback_path = "/opt/vgg16/RecommandSystem_py/model/best.pt"
    if os.path.exists(fallback_path):
        yolo_model = YOLO(fallback_path).to(device)
        logger.info(f"YOLO model loaded via fallback path: {fallback_path}")
    else:
        yolo_model = None
        logger.warning(f"YOLO model NOT FOUND. Checked: {YOLO_MODEL_PATH} and {fallback_path}")

app = FastAPI()

@app.post("/visualize/")
async def visualize_image(file: UploadFile = File(...)):
    if yolo_model is None:
        return JSONResponse(status_code=500, content={"error": "YOLO model not loaded. Check model path."})

    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")

        results = yolo_model.predict(np.array(image), conf=YOLO_CONF_THRESHOLD, verbose=False)
        boxes = results[0].boxes

        detections = []
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                b = box.xyxy[0].cpu().numpy()
                detections.append({
                    "class": CLASS_NAMES[int(box.cls[0].item())] if int(box.cls[0].item()) < len(CLASS_NAMES) else "unknown",
                    "confidence": round(box.conf[0].item(), 4),
                    "coordinate": [int(b[0]), int(b[1]), int(b[2]), int(b[3])]
                })

        return JSONResponse(content={"detections": detections})
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(BytesIO(image_data)).convert("RGB")

    cropped_image, detectedClass, confidence, coordinate, all_detections = detect_and_crop(image)

    img_tensor = load_and_preprocess_image(cropped_image)

    features = extract_features(img_tensor)
    order = extract_feature_means_sort(img_tensor)

    features_base64 = base64.b64encode(features).decode("utf-8")

    return JSONResponse(content={
        "order": order,
        "featuresBase64": features_base64,
        "detectedClass": detectedClass,
        "confidence": round(confidence, 4) if confidence is not None else None,
        "coordinate": coordinate,
        "detections": all_detections
    })

@app.post("/process-image/crop/")
async def process_image_crop(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(BytesIO(image_data)).convert("RGB")

    img_tensor = load_and_preprocess_image(image)

    features = extract_features(img_tensor)
    order = extract_feature_means_sort(img_tensor)

    features_base64 = base64.b64encode(features).decode("utf-8")

    return JSONResponse(content={
        "order": order,
        "featuresBase64": features_base64
    })

def detect_and_crop(image: Image.Image):
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
            "coordinate": [int(max(0, b[0])), int(max(0, b[1])), int(min(w, b[2])), int(min(h, b[3]))]
        })

    best_idx = boxes.conf.argmax().item()
    best_box = boxes.xyxy[best_idx].cpu().numpy().astype(int)
    best_conf = boxes.conf[best_idx].item()
    best_cls = int(boxes.cls[best_idx].item())
    class_name = CLASS_NAMES[best_cls] if best_cls < len(CLASS_NAMES) else "unknown"

    if len(all_detections) > best_idx:
        all_detections.pop(best_idx)

    x1, y1 = max(0, best_box[0]), max(0, best_box[1])
    x2, y2 = min(w, best_box[2]), min(h, best_box[3])
    cropped = image.crop((x1, y1, x2, y2))

    if cropped.size[0] == 0 or cropped.size[1] == 0:
        return image, None, None, None, all_detections

    coordinate = [int(x1), int(y1), int(x2), int(y2)]
    return cropped, class_name, best_conf, coordinate, all_detections

def load_and_preprocess_image(image: Image.Image):
    return preprocess(image.convert("RGB")).unsqueeze(0).to(device)

def extract_feature_means_sort(img_tensor):
    with torch.no_grad():
        feature_maps = model_mid(img_tensor).squeeze(0).cpu().numpy()
    
    feature_dict = {
        f"{i}": float(np.mean(feature_maps[i, :, :])) for i in range(512)
    }

    sorted_features = sorted(feature_dict.items(), key=lambda x: x[1], reverse=True)
    layer_numbers = [int(layer) for layer, _ in sorted_features[:25]]
    return json.dumps(layer_numbers)

def extract_features(img_tensor):
    with torch.no_grad():
        features = base_extractor(img_tensor).flatten().cpu().numpy()
    
    nonzero_features = features[features != 0]
    mean_value = np.mean(nonzero_features) if len(nonzero_features) > 0 else 0
    binary_features = np.where(features >= mean_value, 1, 0)
    return np.packbits(binary_features).tobytes()

app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
