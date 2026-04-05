"""
배치 특징점 추출 스크립트 (PyTorch 버전 - TensorFlow 제거)
- images_*.csv 에서 이미지 목록 읽기
- YOLO로 상품 감지 후 crop
- VGG16 feature 추출 (binary + order) — ONNX Runtime 배치 inference
- CSV로 저장 (500개 단위 분할)
"""

import json
import os
import csv
import glob
import logging
import base64
import time

import numpy as np
import onnxruntime as ort
from PIL import Image
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── 절대 경로 설정 (환경변수 우선, 기본값은 Docker 경로) ──
MODEL_DIR = os.getenv("MODEL_DIR", "/app/model")
CLASSIFY_DIR = os.getenv("CLASSIFY_DIR", "/app/classify")
IMG_DIR = os.getenv("IMG_DIR", "/app/upload")

# ── YOLO 모델 로드 ──
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
YOLO_CONF_THRESHOLD = 0.5
CLASS_NAMES = ["bag", "sunglasses", "food_drink", "shoes", "clothing"]

if os.path.exists(YOLO_MODEL_PATH):
    yolo_model = YOLO(YOLO_MODEL_PATH).to(device)
    logger.info(f"YOLO 모델 로드 완료: {YOLO_MODEL_PATH}")
else:
    logger.warning(f"YOLO 모델을 찾을 수 없습니다: {YOLO_MODEL_PATH}")
    yolo_model = None

CSV_DIR = os.path.join(CLASSIFY_DIR, "csv")
JSON_DIR = os.path.join(CLASSIFY_DIR, "json")

BATCH_SIZE = 32
SPLIT_SIZE = 500

# ── VGG16 ONNX 모델 로드 ──
ONNX_FULL_PATH = os.path.join(MODEL_DIR, "vgg16_full.onnx")
ONNX_BLOCK5_PATH = os.path.join(MODEL_DIR, "vgg16_block5_conv3.onnx")

logger.info("VGG16 ONNX 모델 로딩...")
session_full = ort.InferenceSession(ONNX_FULL_PATH)
session_block5 = ort.InferenceSession(ONNX_BLOCK5_PATH)
logger.info("VGG16 ONNX 모델 로딩 완료")


# ── VGG16 전처리 (TensorFlow preprocess_input 동일) ──
def preprocess_input_vgg16(img: np.ndarray) -> np.ndarray:
    """RGB→BGR 변환 후 ImageNet 채널별 평균 차감 (TF caffe 모드와 동일)"""
    img = img[..., ::-1].copy()  # RGB → BGR
    img[..., 0] -= 103.939
    img[..., 1] -= 116.779
    img[..., 2] -= 123.68
    return img

# Keras VGG16 호환 이미지 전처리
def load_and_preprocess(image: Image.Image) -> torch.Tensor:
    img = image.convert("RGB").resize((224, 224))
    img = preprocess_input_vgg16(np.array(img, dtype=np.float32))
    return img

BATCH_SIZE = 32
SPLIT_SIZE = 500

def extract_features_batch(imgs: np.ndarray) -> list[bytes]:
    input_name = session_full.get_inputs()[0].name
    preds = session_full.run(None, {input_name: imgs})[0]
    results = []
    for i in range(preds.shape[0]):
        features = preds[i].flatten()
        nonzero = features[features != 0]
        mean_val = np.mean(nonzero) if len(nonzero) > 0 else 0
        binary = np.where(features >= mean_val, 1, 0)
        results.append(np.packbits(binary).tobytes())
    return results


def extract_feature_order_batch(imgs: np.ndarray) -> list[str]:
    input_name = session_block5.get_inputs()[0].name
    fmaps = session_block5.run(None, {input_name: imgs})[0]
    results = []
    for i in range(fmaps.shape[0]):
        fm = fmaps[i]
        means = np.mean(fm, axis=(0, 1))
        top25 = (np.argsort(means)[::-1][:25]).tolist()
        results.append(json.dumps(top25))
    return results

def find_image_path(image_url: str) -> str | None:
    """image_url (예: /upload/가방/xxx.jpg) → 실제 파일 경로"""
    rel_path = image_url.replace("/upload/", "").lstrip("/")
    img_path = os.path.join(IMG_DIR, rel_path)
    if os.path.exists(img_path):
        return img_path
    return None

def crop_image_with_yolo(image: Image.Image) -> Image.Image:
    if yolo_model is None:
        return image

    results = yolo_model.predict(np.array(image), conf=YOLO_CONF_THRESHOLD, verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return image

    best_idx = boxes.conf.argmax().item()
    best_box = boxes.xyxy[best_idx].cpu().numpy().astype(int)

    w, h = image.size
    x1, y1 = max(0, best_box[0]), max(0, best_box[1])
    x2, y2 = min(w, best_box[2]), min(h, best_box[3])

    if x2 <= x1 or y2 <= y1:
        return image

    return image.crop((x1, y1, x2, y2))

def load_image_list() -> list[dict]:
    rows = []
    csv_files = sorted(glob.glob(os.path.join(CSV_DIR, "images_*.csv")))
    logger.info(f"images CSV 파일 수: {len(csv_files)}")

    for csv_path in csv_files:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

    logger.info(f"전체 이미지 수: {len(rows)}")
    return rows

def get_existing_file_index() -> int:
    existing = glob.glob(os.path.join(CSV_DIR, "features_*.csv"))
    if not existing:
        return 0
    nums = []
    for f in existing:
        name = os.path.splitext(os.path.basename(f))[0]
        try:
            nums.append(int(name.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return max(nums) if nums else 0

def get_processed_names() -> set:
    processed = set()
    for csv_path in glob.glob(os.path.join(CSV_DIR, "features_*.csv")):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    processed.add(row[0])
    return processed

def main():
    image_list = load_image_list()
    total = len(image_list)

    processed = get_processed_names()
    logger.info(f"이미 처리된 항목: {len(processed)}건")

    file_index = get_existing_file_index()
    success = 0
    skipped = 0
    failed = 0
    failed_list = []

    output_rows = []
    batch_tensors = []
    batch_meta = []

    start_time = time.time()

    def save_csv_chunk():
        nonlocal file_index, output_rows
        while len(output_rows) >= SPLIT_SIZE:
            chunk = output_rows[:SPLIT_SIZE]
            output_rows = output_rows[SPLIT_SIZE:]
            file_index += 1
            out_path = os.path.join(CSV_DIR, f"features_{file_index:03d}.csv")
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["image_uuid", "img_feature_value_base64", "img_feature_order"])
                writer.writerows(chunk)
            logger.info(f"저장: {out_path} ({len(chunk)}건)")

    def flush_batch():
        nonlocal success
        if not batch_tensors:
            return

        imgs_tensor = torch.stack(batch_tensors, dim=0).to(device)

        feat_bytes_list = extract_features_batch(imgs_tensor)
        feat_order_list = extract_feature_order_batch(imgs_tensor)

        for meta, feat_bytes, feat_order in zip(batch_meta, feat_bytes_list, feat_order_list):
            feat_b64 = base64.b64encode(feat_bytes).decode("utf-8")
            output_rows.append([meta, feat_b64, feat_order])
            success += 1

        batch_tensors.clear()
        batch_meta.clear()
        save_csv_chunk()

    try:
        for idx, row in enumerate(image_list):
            image_name = row["image_original_name"]
            image_uuid = row["image_uuid"]
            image_url = row["image_url"]

            if image_uuid in processed:
                skipped += 1
                continue

            try:
                img_path = find_image_path(image_url)
                if img_path is None:
                    failed += 1
                    failed_list.append(f"{image_name} - 이미지 파일 없음")
                    continue

                image = Image.open(img_path)

                # YOLO를 이용한 가변 Crop
                cropped = crop_image_with_yolo(image)
                preprocessed = load_and_preprocess(cropped)
                
                batch_tensors.append(preprocessed)
                batch_meta.append(image_uuid)

                if len(batch_tensors) >= BATCH_SIZE:
                    flush_batch()

            except Exception as e:
                failed += 1
                failed_list.append(f"{image_name} - {str(e)}")

            done = success + skipped + failed
            if done > 0 and done % (BATCH_SIZE * 10) == 0:
                elapsed = time.time() - start_time
                speed = success / elapsed if elapsed > 0 else 0
                logger.info(
                    f"진행: {done}/{total} (성공: {success}, 스킵: {skipped}, 실패: {failed}) "
                    f"| {speed:.1f} img/s"
                )

        flush_batch()

        if output_rows:
            file_index += 1
            out_path = os.path.join(CSV_DIR, f"features_{file_index:03d}.csv")
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["image_uuid", "img_feature_value_base64", "img_feature_order"])
                writer.writerows(output_rows)
            logger.info(f"저장: {out_path} ({len(output_rows)}건)")
            output_rows.clear()

    finally:
        pass

    elapsed = time.time() - start_time

    if failed_list:
        fail_log = os.path.join(CSV_DIR, "feature_extract_failed.log")
        with open(fail_log, "w", encoding="utf-8") as f:
            for line in failed_list:
                f.write(line + "\n")
        logger.warning(f"실패 목록 저장: {fail_log} ({len(failed_list)}건)")

    logger.info(
        f"완료! 성공: {success}, 스킵: {skipped}, 실패: {failed} "
        f"| 소요시간: {elapsed:.1f}s ({success / elapsed:.1f} img/s)"
    )

if __name__ == "__main__":
    main()
