"""
배치 특징점 추출 스크립트 — ResNet-50 + PQ (멀티프로세스 병렬 처리).

  - images_*.csv 에서 이미지 목록 읽기
  - YOLO 로 상품 감지 + crop + detected_class 보존
  - ResNet-50 + GAP → 2048D float 임베딩
      • embedding_base64 : fp16 × 2048 = 4,096 byte  (Phase 2 코사인 재정렬용)
      • pq_code_base64   : PQ 압축 → 64 byte         (Phase 1 1차 필터용)
      • detected_class   : YOLO 분류 (예: bag/shoes) (Level 1 파티션용)
  - NUM_WORKERS 프로세스 병렬 처리
  - CSV 500개 단위 분할 저장 → resnet50_NNN.csv

실행:
  python workFile/batch_extract_resnet50.py
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

from api.resnet50 import ResNet50Backbone
from config import (
    CLASS_NAMES,
    CLASSIFY_DIR,
    IMG_DIR,
    KOREAN_TO_EN,
    MODEL_DIR,
    NUM_WORKERS,
    YOLO_CONF_THRESHOLD,
)

SPLIT_SIZE   = 500
CSV_DIR      = os.path.join(CLASSIFY_DIR, "csv")
OUTPUT_PREFIX = "resnet50"
CSV_HEADER   = ["image_uuid", "embedding_base64", "pq_code_base64", "detected_class"]


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
_w_backbone: ResNet50Backbone | None = None
_w_yolo:     YOLO | None = None


def _worker_init(model_dir: str):
    """각 워커 프로세스 시작 시 1회 실행 — 모델 + 코드북 로드."""
    global _w_backbone, _w_yolo

    _w_backbone = ResNet50Backbone(model_dir)
    if not _w_backbone.is_loaded():
        raise RuntimeError(
            f"ResNet-50 ONNX 모델을 찾을 수 없습니다: {model_dir}. "
            f"workFile/export_resnet50.py 먼저 실행 필요."
        )
    if _w_backbone._codebook is None:
        raise RuntimeError(
            f"PQ 코드북을 찾을 수 없습니다: {model_dir}/pq_codebook.npy. "
            f"workFile/train_pq_codebook.py 먼저 실행 필요."
        )

    yolo_path = os.path.join(model_dir, "best.pt")
    _w_yolo = YOLO(yolo_path) if os.path.exists(yolo_path) else None


def _crop_with_yolo(image: Image.Image) -> tuple[Image.Image, str | None]:
    """
    YOLO 로 confidence 최고 박스만 crop + 클래스 반환.
    미로드/미검출 시 (원본, None) 반환.
    """
    if _w_yolo is None:
        return image, None

    results = _w_yolo.predict(np.array(image), conf=YOLO_CONF_THRESHOLD, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return image, None

    best_idx = boxes.conf.argmax().item()
    best_box = boxes.xyxy[best_idx].cpu().numpy().astype(int)
    best_cls = int(boxes.cls[best_idx].item())
    class_name = CLASS_NAMES[best_cls] if best_cls < len(CLASS_NAMES) else "unknown"

    w, h = image.size
    x1, y1 = max(0, best_box[0]), max(0, best_box[1])
    x2, y2 = min(w, best_box[2]), min(h, best_box[3])
    if x2 <= x1 or y2 <= y1:
        return image, class_name

    return image.crop((x1, y1, x2, y2)), class_name


def _worker_process(row: dict) -> dict:
    """이미지 1장 처리 — 워커 프로세스에서 실행."""
    image_uuid = row["image_uuid"]
    image_url  = row["image_url"]
    image_name = row.get("image_original_name", image_uuid)

    img_path = _resolve_image_path(image_url)
    if not os.path.exists(img_path):
        return {"status": "failed", "uuid": image_uuid, "name": image_name, "reason": "이미지 파일 없음"}

    try:
        image = Image.open(img_path).convert("RGB")
        cropped, det_class = _crop_with_yolo(image)
        output = _w_backbone.extract(cropped)

        # ResNet50Backbone 은 코드북이 있으면 항상 pq_code_bytes 채움.
        # _worker_init 에서 코드북 존재 보장됨.
        if output.embedding_bytes is None or output.pq_code_bytes is None:
            return {
                "status": "failed", "uuid": image_uuid, "name": image_name,
                "reason": "embedding/pq_code 누락 — 코드북 미로드 가능성",
            }

        return {
            "status":   "ok",
            "uuid":     image_uuid,
            "emb_b64":  base64.b64encode(output.embedding_bytes).decode(),
            "pq_b64":   base64.b64encode(output.pq_code_bytes).decode(),
            "det_cls":  det_class or "",  # 미검출 시 빈 문자열
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
    existing = glob.glob(os.path.join(CSV_DIR, f"{OUTPUT_PREFIX}_*.csv"))
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
    for csv_path in glob.glob(os.path.join(CSV_DIR, f"{OUTPUT_PREFIX}_*.csv")):
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

    # 사전 검증 — 코드북 + ONNX 존재 확인 (워커 init 전에 실패 표면화)
    codebook_path = os.path.join(MODEL_DIR, "pq_codebook.npy")
    onnx_path     = os.path.join(MODEL_DIR, "resnet50.onnx")
    if not os.path.exists(onnx_path):
        raise RuntimeError(f"ResNet-50 ONNX 없음: {onnx_path}")
    if not os.path.exists(codebook_path):
        raise RuntimeError(f"PQ 코드북 없음: {codebook_path}")

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
            out_path = os.path.join(CSV_DIR, f"{OUTPUT_PREFIX}_{file_index:03d}.csv")
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADER)
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
                    result["emb_b64"],
                    result["pq_b64"],
                    result["det_cls"],
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
        out_path = os.path.join(CSV_DIR, f"{OUTPUT_PREFIX}_{file_index:03d}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
            writer.writerows(output_rows)
        logger.info(f"저장: {out_path} ({len(output_rows)}건)")

    elapsed = time.time() - start_time

    if failed_list:
        fail_log = os.path.join(CSV_DIR, "resnet50_extract_failed.log")
        with open(fail_log, "w", encoding="utf-8") as f:
            f.writelines(line + "\n" for line in failed_list)
        logger.warning(f"실패 목록: {fail_log} ({len(failed_list)}건)")

    logger.info(
        f"완료! 성공: {success}, 스킵: {skipped}, 실패: {failed} "
        f"| {elapsed:.1f}s ({success / elapsed if elapsed > 0 else 0:.1f} img/s)"
    )


if __name__ == "__main__":
    mp.freeze_support()  # Windows 필수
    main()
