import json
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import base64

# 1. VGG16 모델 (conv5_block3_conv3 레이어까지 가져오기)
base_model = VGG16(weights="imagenet", include_top=False)
layer_name = "block5_conv3"  # 마지막 Conv Layer
model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    image_data = await file.read()
    np_image = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    img = load_and_preprocess_image(image)  # 변환된 NumPy 배열 사용

    features = extract_features(img)  # byte[] , 이미지의 특징점
    order = extract_feature_means_sort(img)  # 문자열, 크기순으로 정렬된 레이어 번호

    # 🔥 features를 Base64 문자열로 변환
    features_base64 = base64.b64encode(features).decode("utf-8")

    return JSONResponse(content={"order": order, "features": features_base64})

# 2. 이미지 전처리 함수 (파일 경로 대신 PIL 이미지를 받도록 수정)
def load_and_preprocess_image(image: Image.Image):
    img = np.array(image)  # PIL 이미지를 NumPy 배열로 변환
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB → BGR 변환 (OpenCV 사용)
    img = cv2.resize(img, (224, 224))  # 모델 입력 크기에 맞게 조정
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


# 3. 각 레이어의 Intensity 값이 큰 순서대로 정렬
def extract_feature_means_sort(img):

    feature_maps = model.predict(img)  # (1, 14, 14, 512) 형태
    feature_maps = feature_maps.squeeze()  # (14, 14, 512)로 변환

    # 각 레이어의 평균 Intensity 계산
    feature_dict = {
        f"{i + 1}": np.mean(feature_maps[:, :, i]) for i in range(512)
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
