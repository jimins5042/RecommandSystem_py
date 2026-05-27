"""
프로젝트 전체 설정 — 경로, 환경변수, 상수.
web/ 와 workFile/ 에서 공통으로 import.

우선순위 (높음→낮음):
  1. 프로세스 환경변수 (Dockerfile ENV)
  2. .env 파일
  3. 프로젝트 상대경로 fallback
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# ── 서빙 ──
MODEL_DIR  = os.getenv("MODEL_DIR",  str(PROJECT_ROOT / "model"))
STATIC_DIR = os.getenv("STATIC_DIR", str(PROJECT_ROOT / "static"))
YOLO_CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.5"))

# ── 도메인 상수 ──
CLASS_NAMES = ["bag", "sunglasses", "food_drink", "shoes", "clothing"]

# 한글 카테고리 → 디스크 폴더명 매핑 (Spring 측 upload 폴더 구조 기준)
# 주의: CLASS_NAMES (YOLO 학습 라벨) 와는 별개 네임스페이스.
#       - 디스크 폴더: bag / sunglass / food / shoes / clothes
#       - YOLO 출력 (detected_class): bag / sunglasses / food_drink / shoes / clothing
KOREAN_TO_EN = {
    "가방":   "bag",
    "선글라스": "sunglass",
    "식음료":  "food",
    "신발":   "shoes",
    "의류":   "clothes",
}

# ── 배치 추출 ──
CLASSIFY_DIR = os.getenv("CLASSIFY_DIR", str(PROJECT_ROOT / "classify"))
IMG_DIR      = os.getenv("IMG_DIR",      str(PROJECT_ROOT / "upload"))
NUM_WORKERS  = int(os.getenv("NUM_WORKERS", "6"))

# ── 전처리 (LabelMe → YOLO) ──
SOURCE_DIR = Path(os.getenv("SOURCE_DIR", str(PROJECT_ROOT / "labelme_source")))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "yolo_dataset")))
