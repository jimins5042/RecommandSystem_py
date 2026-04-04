import json
import os
import logging
from io import BytesIO

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from ultralytics import YOLO
import base64

logger = logging.getLogger(__name__)

# 1. VGG16 모델 (conv5_block3_conv3 레이어까지 가져오기)
base_model = VGG16(weights="imagenet", include_top=False)
layer_name = "block5_conv3"  # 마지막 Conv Layer
model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)

# 2. YOLO 모델 (상품 객체 인식)
# 실행 환경에 구애받지 않도록 상대 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "model", "best.pt")
YOLO_CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.5"))
CLASS_NAMES = ["bag", "sunglasses", "food_drink", "shoes", "clothing"]

if os.path.exists(YOLO_MODEL_PATH):
    yolo_model = YOLO(YOLO_MODEL_PATH)
    logger.info(f"YOLO model loaded from {YOLO_MODEL_PATH}")
else:
    yolo_model = None
    logger.warning(f"YOLO model not found at {YOLO_MODEL_PATH}. Running without object detection.")

app = FastAPI()

@app.post("/visualize/")
async def visualize_image(file: UploadFile = File(...)):
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

@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(BytesIO(image_data)).convert("RGB")

    # YOLO 객체 인식 → crop
    cropped_image, detectedClass, confidence, coordinate, all_detections = detect_and_crop(image)

    img = load_and_preprocess_image(cropped_image)

    features = extract_features(img)  # byte[] , 이미지의 특징점
    order = extract_feature_means_sort(img)  # 문자열, 크기순으로 정렬된 레이어 번호

    # features를 Base64 문자열로 변환
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

    # 이미 크롭된 이미지이므로 YOLO 감지 과정을 생략하여 속도 향상
    img = load_and_preprocess_image(image)

    features = extract_features(img)
    order = extract_feature_means_sort(img)

    # features를 Base64 문자열로 변환
    features_base64 = base64.b64encode(features).decode("utf-8")

    return JSONResponse(content={
        "order": order,
        "featuresBase64": features_base64
    })


@app.post("/process-image/crop/")
async def process_cropImage(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(BytesIO(image_data)).convert("RGB")
    img = load_and_preprocess_image(image)

    features = extract_features(img)  # byte[] , 이미지의 특징점
    order = extract_feature_means_sort(img)  # 문자열, 크기순으로 정렬된 레이어 번호

    # features를 Base64 문자열로 변환
    features_base64 = base64.b64encode(features).decode("utf-8")

    return JSONResponse(content={
        "order": order,
        "featuresBase64": features_base64,
    })

def detect_and_crop(image: Image.Image):
    """YOLO로 상품 감지 후 가장 높은 confidence의 객체를 crop하고, 전체 감지 목록도 반환"""
    if yolo_model is None:
        return image, None, None, None, []

    results = yolo_model.predict(np.array(image), conf=YOLO_CONF_THRESHOLD, verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return image, None, None, None, []

    # 전체 감지 결과
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

    # confidence 가장 높은 1개 선택
    best_idx = boxes.conf.argmax().item()
    best_box = boxes.xyxy[best_idx].cpu().numpy().astype(int)
    best_conf = boxes.conf[best_idx].item()
    best_cls = int(boxes.cls[best_idx].item())
    class_name = CLASS_NAMES[best_cls] if best_cls < len(CLASS_NAMES) else "unknown"

    # crop (경계 클리핑)
    x1, y1 = max(0, best_box[0]), max(0, best_box[1])
    x2, y2 = min(w, best_box[2]), min(h, best_box[3])
    cropped = image.crop((x1, y1, x2, y2))

    if cropped.size[0] == 0 or cropped.size[1] == 0:
        return image, None, None, None, all_detections

    coordinate = [int(x1), int(y1), int(x2), int(y2)]
    return cropped, class_name, best_conf, coordinate, all_detections


# 2. 이미지 전처리 함수
def load_and_preprocess_image(image: Image.Image):
    img = image.convert("RGB").resize((224, 224))
    img = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    img = preprocess_input(img)
    return img


# 3. 각 레이어의 Intensity 값이 큰 순서대로 정렬
def extract_feature_means_sort(img):

    feature_maps = model.predict(img)  # (1, 14, 14, 512) 형태
    feature_maps = feature_maps.squeeze()  # (14, 14, 512)로 변환

    # 각 레이어의 평균 Intensity 계산
    feature_dict = {
        # f"{i + 1}": np.mean(feature_maps[:, :, i]) for i in range(512)
        f"{i}": np.mean(feature_maps[:, :, i]) for i in range(512)
    }

    # 내림차순 정렬 후 상위 25개 선택
    sorted_features = sorted(feature_dict.items(), key=lambda x: x[1], reverse=True)
    layer_numbers = [int(layer) for layer, _ in sorted_features[:25]]
    json_data = json.dumps(layer_numbers)
    print(json_data)

    return json.dumps(layer_numbers)  # JSON 문자열 변환 후 반환

# 4. 이미지의 특징점 추출 및 이진화
def extract_features(img):

    features = base_model.predict(img).flatten()  # 1D 벡터 변환

    # 0을 제외한 값들의 평균 계산
    nonzero_features = features[features != 0]
    mean_value = np.mean(nonzero_features) if len(nonzero_features) > 0 else 0

    # 이진화: 평균 이상이면 1, 미만이면 0
    binary_features = np.where(features >= mean_value, 1, 0)
    binary_bytes = np.packbits(binary_features).tobytes()  # NumPy 배열을 bytes로 변환
    #print(f" 이진화된 특징점 일부: {binary_features[:100]}")
    return binary_bytes

# static 파일 서빙 (API 라우트 등록 후 마운트해야 가려지지 않음)
app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
