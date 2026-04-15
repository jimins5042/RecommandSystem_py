"""
배치 특징점 추출 스크립트 (멀티프로세스 병렬 처리)
- images_*.csv 에서 이미지 목록 읽기
- YOLO로 상품 감지 후 crop
- EfficientNet-B0 feature 추출 — backbones/ 패키지 재사용
  Stage 1 (order)         : 7x7x320 → GAP → Top-25 인덱스
  Stage 2 (featuresBase64): 1280D  → 평균 이진화 → 160 byte
  Stage 3 (embedding)     : 1280D  → float16 → 2560 byte (Cosine 재정렬용)
- NUM_WORKERS 프로세스 병렬 처리
- CSV로 저장 (500개 단위 분할)

실행:
  python workFile/batch_extract.py       # 프로젝트 루트에서
"""
from __future__ import annotations

# ── 프로젝트 루트를 sys.path 에 추가 (스크립트 직접 실행 지원) ──
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import base64
import csv
import glob
import logging
import multiprocessing as mp
import os
import time

import numpy as np
from PIL import Image
from ultralytics import YOLO

from config import (
    CLASSIFY_DIR,
    IMG_DIR,
    KOREAN_TO_EN,
    MODEL_DIR,
    NUM_WORKERS,
    YOLO_CONF_THRESHOLD,
)
from api.efficientnet_b0 import EfficientNetB0Backbone

SPLIT_SIZE = 500
CSV_DIR    = os.path.join(CLASSIFY_DIR, "csv")


# ════════════════════════════════════════
# 경로 해석
# ════════════════════════════════════════

def _resolve_image_path(image_url: str) -> str:
    """image_url (예: /upload/가방/xxx.jpg) → 실제 로컬 파일 경로."""
    rel_path = image_url.replace("/upload/", "").lstrip("/")
    for ko, en in KOREAN_TO_EN.items():
        rel_path = rel_path.replace(ko, en)
    return os.path.join(IMG_DIR, rel_path)


# ════════════════════════════════════════
# 워커 프로세스
# ════════════════════════════════════════

# 워커별 전역 (initializer 에서 세팅). fork 시 복제 방지 위해 None 시작.
_w_backbone: EfficientNetB0Backbone | None = None
_w_yolo:     YOLO | None = None


def _worker_init(model_dir: str):
    """각 워커 프로세스 시작 시 1회 실행 — 모델 로드."""
    global _w_backbone, _w_yolo

    _w_backbone = EfficientNetB0Backbone(model_dir)
    if not _w_backbone.is_loaded():
        raise RuntimeError(f"EfficientNet-B0 ONNX 모델을 찾을 수 없습니다: {model_dir}")

    yolo_path = os.path.join(model_dir, "best.pt")
    _w_yolo = YOLO(yolo_path) if os.path.exists(yolo_path) else None


def _crop_with_yolo(image: Image.Image) -> Image.Image:
    """YOLO 로 confidence 최고 박스만 crop. 미로드/미검출 시 원본."""
    if _w_yolo is None:
        return image
    results = _w_yolo.predict(np.array(image), conf=YOLO_CONF_THRESHOLD, verbose=False)
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


def _worker_process(row: dict) -> dict:
    """이미지 1장 처리 — 워커 프로세스에서 실행."""
    image_uuid = row["image_uuid"]
    image_url  = row["image_url"]
    image_name = row.get("image_original_name", image_uuid)

    img_path = _resolve_image_path(image_url)
    if not os.path.exists(img_path):
        return {"status": "failed", "uuid": image_uuid, "name": image_name, "reason": "이미지 파일 없음"}

    try:
        image   = Image.open(img_path).convert("RGB")
        cropped = _crop_with_yolo(image)
        output  = _w_backbone.extract(cropped)

        return {
            "status":   "ok",
            "uuid":     image_uuid,
            "feat_b64": base64.b64encode(output.features_bytes).decode(),
            "order":    output.order,
            "emb_b64":  base64.b64encode(output.embedding_bytes or b"").decode(),
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
    success = failed = 0
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
        initargs=(MODEL_DIR,),
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
