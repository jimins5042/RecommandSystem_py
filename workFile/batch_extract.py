"""
배치 특징점 추출 스크립트 (멀티프로세스 병렬 처리)
- images_*.csv 에서 이미지 목록 읽기
- YOLO로 상품 감지 후 crop
- EfficientNet-B0 feature 추출 (binary + order + embedding)
  Stage 1 (order)         : 7x7x320 → GAP → Top-25 인덱스
  Stage 2 (featuresBase64): 1280D  → 평균 이진화 → 160 byte
  Stage 3 (embedding)     : 1280D  → float16 → 2560 byte (Cosine 재정렬용)
- NUM_WORKERS 프로세스 병렬 처리
- CSV로 저장 (500개 단위 분할)
"""

import json
import os
import csv
import glob
import logging
import base64
import time
import multiprocessing as mp

import numpy as np
import onnxruntime as ort
from PIL import Image
from ultralytics import YOLO

# ── 설정 ──
CLASSIFY_DIR = os.getenv("CLASSIFY_DIR", "C:\\Users\\coolc\\OneDrive\\Desktop\\marqvision\\classify")
IMG_DIR      = os.getenv("IMG_DIR",      "C:\\Users\\coolc\\IdeaProjects\\RecommendSystem\\src\\main\\resources\\upload")

MODEL_DIR = os.getenv("MODEL_DIR", "C:\\Users\\coolc\\PycharmProjects\\recommandSystem-py\\model")
STATIC_DIR = os.getenv("STATIC_DIR", "C:\\Users\\coolc\\PycharmProjects\\recommandSystem-py\\static")

NUM_WORKERS  = int(os.getenv("NUM_WORKERS", "6"))

YOLO_CONF_THRESHOLD = 0.5
SPLIT_SIZE          = 500

CSV_DIR  = os.path.join(CLASSIFY_DIR, "csv")

EFFICIENTNET_PATH   = os.path.join(MODEL_DIR, "efficientnet-b0-feat.onnx")
EFFICIENTNET_STAGE1 = "/features/features.7/features.7.0/block/block.3/block.3.0/Conv_output_0"
EFFICIENTNET_STAGE2 = "/Flatten_output_0"

# ── EfficientNet-B0 전처리 상수 (모든 프로세스에서 공유) ──
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


# ════════════════════════════════════════
# 공유 유틸
# ════════════════════════════════════════

def load_and_preprocess(image: Image.Image) -> np.ndarray:
    """PIL Image → CHW float32 [3,224,224], ImageNet 정규화"""
    img = np.array(image.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return ((img - _MEAN) / _STD).astype(np.float32)


def extract_single(stage1_map: np.ndarray, emb: np.ndarray) -> tuple[str, bytes, bytes]:
    """
    추론 결과 1장을 Stage1/2/3 feature로 변환.
    stage1_map : (320, 7, 7)
    emb        : (1280,)
    """
    channel_scores = stage1_map.mean(axis=(1, 2))
    top25 = np.argsort(channel_scores)[::-1][:25].tolist()

    nonzero = emb[emb != 0]
    mean_val = np.mean(nonzero) if len(nonzero) > 0 else 0.0
    binary = np.where(emb >= mean_val, 1, 0)
    feature_bytes = np.packbits(binary).tobytes()

    embedding_bytes = emb.astype(np.float16).tobytes()

    return json.dumps(top25), feature_bytes, embedding_bytes


# ════════════════════════════════════════
# 워커 프로세스
# ════════════════════════════════════════

# 워커별 전역 (initializer에서 세팅)
_w_session = None
_w_yolo    = None


def _worker_init(model_dir: str, img_dir: str):
    """각 워커 프로세스 시작 시 1회 실행 — 모델 로드"""
    global _w_session, _w_yolo, IMG_DIR

    IMG_DIR = img_dir

    _w_session = ort.InferenceSession(os.path.join(model_dir, "efficientnet-b0-feat.onnx"))

    yolo_path = os.path.join(model_dir, "best.pt")
    if os.path.exists(yolo_path):
        _w_yolo = YOLO(yolo_path)


def _worker_process(row: dict) -> dict:
    """이미지 1장 처리 — 워커 프로세스에서 실행"""
    image_uuid = row["image_uuid"]
    image_url  = row["image_url"]
    image_name = row.get("image_original_name", image_uuid)

    rel_path = (image_url.replace("/upload/", "").lstrip("/").replace("가방", "bag").replace("선글라스", "sunglass").replace("식음료", "food").replace("신발", "shoes").replace("의류", "clothes"))  
    img_path = os.path.join(IMG_DIR, rel_path)
    print(img_path)

    if not os.path.exists(img_path):
        return {"status": "failed", "uuid": image_uuid, "name": image_name, "reason": "이미지 파일 없음"}

    try:
        image = Image.open(img_path).convert("RGB")

        # YOLO crop
        if _w_yolo is not None:
            results = _w_yolo.predict(np.array(image), conf=YOLO_CONF_THRESHOLD, verbose=False)
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                best_idx = boxes.conf.argmax().item()
                best_box = boxes.xyxy[best_idx].cpu().numpy().astype(int)
                w, h = image.size
                x1 = max(0, best_box[0]);  y1 = max(0, best_box[1])
                x2 = min(w, best_box[2]);  y2 = min(h, best_box[3])
                if x2 > x1 and y2 > y1:
                    image = image.crop((x1, y1, x2, y2))

        # 전처리 + 추론
        img_array = np.expand_dims(load_and_preprocess(image), axis=0)  # (1,3,224,224)
        input_name = _w_session.get_inputs()[0].name
        stage1_maps, embeddings = _w_session.run(
            [EFFICIENTNET_STAGE1, EFFICIENTNET_STAGE2],
            {input_name: img_array}
        )

        order, feat_bytes, emb_bytes = extract_single(stage1_maps[0], embeddings[0])

        return {
            "status":   "ok",
            "uuid":     image_uuid,
            "feat_b64": base64.b64encode(feat_bytes).decode(),
            "order":    order,
            "emb_b64":  base64.b64encode(emb_bytes).decode(),
        }

    except Exception as e:
        return {"status": "failed", "uuid": image_uuid, "name": image_name, "reason": str(e)}


# ════════════════════════════════════════
# CSV 유틸
# ════════════════════════════════════════

def load_image_list() -> list[dict]:
    rows = []
    for csv_path in sorted(glob.glob(os.path.join(CSV_DIR, "images_*.csv"))):
        with open(csv_path, "r", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def get_existing_file_index() -> int:
    existing = glob.glob(os.path.join(CSV_DIR, "efficientNet_*.csv"))
    if not existing:
        return 0
    nums = []
    for f in existing:
        try:
            nums.append(int(os.path.splitext(os.path.basename(f))[0].split("_")[1]))
        except (IndexError, ValueError):
            pass
    return max(nums) if nums else 0


def get_processed_names() -> set:
    processed = set()
    for csv_path in glob.glob(os.path.join(CSV_DIR, "efficientNet_*.csv")):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    processed.add(row[0])
    return processed


# ════════════════════════════════════════
# 메인
# ════════════════════════════════════════

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    image_list = load_image_list()
    logger.info(f"전체 이미지: {len(image_list)}건")

    processed = get_processed_names()
    logger.info(f"이미 처리됨: {len(processed)}건")

    todo = [r for r in image_list if r["image_uuid"] not in processed]
    logger.info(f"처리 대상: {len(todo)}건 | 워커: {NUM_WORKERS}개")

    file_index  = get_existing_file_index()
    output_rows = []
    success = skipped = failed = 0
    failed_list = []
    start_time = time.time()

    def save_chunk():
        nonlocal file_index, output_rows
        while len(output_rows) >= SPLIT_SIZE:
            chunk = output_rows[:SPLIT_SIZE]
            output_rows = output_rows[SPLIT_SIZE:]
            file_index += 1
            out_path = os.path.join(CSV_DIR, f"efficientNet_{file_index:03d}.csv")
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["image_uuid", "img_feature_value_base64", "img_feature_order", "img_embedding_base64"])
                writer.writerows(chunk)
            logger.info(f"저장: {out_path} ({len(chunk)}건)")

    skipped = len(image_list) - len(todo)

    with mp.Pool(
        processes=NUM_WORKERS,
        initializer=_worker_init,
        initargs=(MODEL_DIR, IMG_DIR),
    ) as pool:
        for result in pool.imap_unordered(_worker_process, todo, chunksize=NUM_WORKERS * 4):
            if result["status"] == "ok":
                output_rows.append([
                    result["uuid"],
                    result["feat_b64"],
                    result["order"],
                    result["emb_b64"],
                ])
                success += 1
            else:
                failed += 1
                failed_list.append(f'{result["name"]} - {result["reason"]}')

            save_chunk()

            done = success + failed
            if done % 100 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"진행: {done}/{len(todo)} (성공: {success}, 실패: {failed}) "
                    f"| {success / elapsed:.1f} img/s"
                )

    # 잔여 저장
    if output_rows:
        file_index += 1
        out_path = os.path.join(CSV_DIR, f"efficientNet_{file_index:03d}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_uuid", "img_feature_value_base64", "img_feature_order", "img_embedding_base64"])
            writer.writerows(output_rows)
        logger.info(f"저장: {out_path} ({len(output_rows)}건)")

    elapsed = time.time() - start_time

    if failed_list:
        fail_log = os.path.join(CSV_DIR, "feature_extract_failed.log")
        with open(fail_log, "w", encoding="utf-8") as f:
            f.writelines(line + "\n" for line in failed_list)
        logger.warning(f"실패 목록: {fail_log} ({len(failed_list)}건)")

    logger.info(
        f"완료! 성공: {success}, 스킵: {skipped}, 실패: {failed} "
        f"| {elapsed:.1f}s ({success / elapsed:.1f} img/s)"
    )


if __name__ == "__main__":
    mp.freeze_support()  # Windows 필수
    main()
