"""
PQ (Product Quantization) 코드북 학습 스크립트.

이미지 디렉토리에서 ResNet-50 임베딩을 추출한 후 FAISS ProductQuantizer로
K-means 학습을 수행하여 코드북 파일을 생성한다.

산출물:
    model/pq_codebook.npy        — 코드북, shape (M=64, K=256, D_sub=32) float32
    model/pq_embeddings_cache.npy — (선택) 추출된 임베딩 캐시 (재학습 시 빠른 재사용)

사용 예:
    # 기본 (yolo_dataset/images/train 에서 5만 장 샘플링)
    python workFile/train_pq_codebook.py

    # 커스텀 디렉토리 + 샘플 수
    python workFile/train_pq_codebook.py --src upload --max-samples 30000

    # 캐시된 임베딩으로 빠르게 재학습 (M/K 파라미터 튜닝 시)
    python workFile/train_pq_codebook.py --use-cache

PQ 파라미터:
    M=64, K=256, D=2048
    → 1상품당 64 byte 압축. 분기/반기 단위 재학습 권장.
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# 프로젝트 루트를 path 에 추가
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from api.resnet50 import ResNet50Backbone
from config import MODEL_DIR
from web.detection import detect_and_crop, yolo_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────
# PQ 하이퍼파라미터
# ─────────────────────────────────────────────────────
EMBEDDING_DIM = 2048       # ResNet-50 출력 차원 (D)
M             = 64         # sub-vector 개수
NBITS         = 8          # sub-vector당 비트 (2^8 = 256 코드북 엔트리)
K             = 1 << NBITS # 256
D_SUB         = EMBEDDING_DIM // M  # 32

# ─────────────────────────────────────────────────────
# 경로
# ─────────────────────────────────────────────────────
CODEBOOK_PATH  = Path(MODEL_DIR) / "pq_codebook.npy"
CACHE_PATH     = Path(MODEL_DIR) / "pq_embeddings_cache.npy"
DEFAULT_SRC    = _PROJECT_ROOT / "yolo_dataset" / "images" / "train"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def collect_image_paths(src_dir: Path, max_samples: int, seed: int) -> list[Path]:
    """src_dir 하위에서 이미지 파일을 재귀 수집 후 무작위 샘플링."""
    if not src_dir.exists():
        raise FileNotFoundError(f"이미지 소스 디렉토리가 존재하지 않음: {src_dir}")

    paths: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(src_dir.rglob(f"*{ext}"))

    if not paths:
        raise FileNotFoundError(f"이미지 파일을 찾지 못함: {src_dir}")

    logger.info(f"수집된 이미지: {len(paths)}장 ({src_dir})")

    rng = random.Random(seed)
    if len(paths) > max_samples:
        paths = rng.sample(paths, max_samples)
        logger.info(f"  → {max_samples}장 샘플링")
    else:
        logger.info(f"  → 전체 {len(paths)}장 사용 (max_samples={max_samples} 미달)")

    return paths


def extract_embeddings(image_paths: list[Path], backbone: ResNet50Backbone) -> np.ndarray:
    """YOLO crop + ResNet-50 임베딩 일괄 추출.

    운영 시점 (web/routes/features.py) 흐름과 동일하게 YOLO crop 을 적용하여
    학습 데이터와 검색 데이터의 분포를 일치시킨다.
    YOLO 감지 실패 시에는 원본 이미지로 fallback (운영 흐름과 동일).
    """
    if yolo_model is None:
        logger.warning(
            "YOLO 모델이 로드되지 않음. crop 없이 원본 이미지로 학습 — "
            "운영 시점과 분포가 어긋날 수 있음."
        )

    n = len(image_paths)
    embeddings = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)
    failed = 0
    crop_fallback = 0  # YOLO 감지 실패 → 원본 사용 카운트

    start = time.time()
    last_log = start

    for i, path in enumerate(image_paths):
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                cropped, det_class, _, _, _ = detect_and_crop(img)
                if det_class is None:
                    crop_fallback += 1
                output = backbone.extract(cropped)

            # embedding_bytes 는 fp16 → float32 복원
            emb = np.frombuffer(output.embedding_bytes, dtype=np.float16).astype(np.float32)
            embeddings[i] = emb
        except Exception as e:
            logger.warning(f"  skip {path.name}: {e}")
            failed += 1

        # 30초 간격 진행률 로깅
        now = time.time()
        if now - last_log > 30 or i + 1 == n:
            elapsed = now - start
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            eta_sec = (n - i - 1) / speed if speed > 0 else 0
            logger.info(
                f"  진행 {i + 1}/{n} ({100 * (i + 1) / n:.1f}%) "
                f"| {speed:.1f} img/s | ETA {eta_sec / 60:.1f}min "
                f"| crop_fallback {crop_fallback}"
            )
            last_log = now

    # 실패한 항목 제거
    if failed > 0:
        valid_mask = np.any(embeddings != 0, axis=1)
        embeddings = embeddings[valid_mask]
        logger.warning(f"실패 {failed}건 제외 → 유효 임베딩 {len(embeddings)}건")

    if crop_fallback > 0:
        logger.info(
            f"YOLO 감지 실패하여 원본 사용: {crop_fallback}/{n}건 "
            f"({100 * crop_fallback / n:.1f}%) — 운영 시 fallback 분포와 동일"
        )

    return embeddings


def train_pq_codebook(embeddings: np.ndarray) -> np.ndarray:
    """FAISS ProductQuantizer 학습 후 코드북 반환.

    반환 shape: (M, K, D_sub) float32 — Spring/FastAPI 측에서 일관되게 사용.
    """
    try:
        import faiss
    except ImportError as e:
        raise ImportError(
            "faiss 가 설치되지 않음. requirements.txt 에 faiss-cpu 가 있는지 확인."
        ) from e

    logger.info(
        f"PQ 학습 시작 — D={EMBEDDING_DIM}, M={M}, K={K}, D_sub={D_SUB} | "
        f"학습 샘플 {len(embeddings)}건"
    )

    pq = faiss.ProductQuantizer(EMBEDDING_DIM, M, NBITS)
    # FAISS 는 contiguous float32 를 요구
    train_data = np.ascontiguousarray(embeddings, dtype=np.float32)

    start = time.time()
    pq.train(train_data)
    elapsed = time.time() - start
    logger.info(f"PQ 학습 완료 — 소요 {elapsed:.1f}s")

    # FAISS 의 centroids 는 (M * K * D_sub,) 1D flat
    codebook = faiss.vector_to_array(pq.centroids).reshape(M, K, D_SUB)
    return codebook


def verify_quantization_quality(embeddings: np.ndarray, codebook: np.ndarray) -> None:
    """인코딩 후 복원한 임베딩과 원본 간 평균 코사인 유사도 측정.

    품질 좋은 PQ 라면 0.9 이상이 나와야 한다.
    """
    # 검증용 샘플 (학습 데이터에서 1000개)
    n_check = min(1000, len(embeddings))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(embeddings), size=n_check, replace=False)
    samples = embeddings[idx]

    # 인코딩: 각 sub-vector 를 가장 가까운 코드북 엔트리로 매핑
    codes = np.zeros((n_check, M), dtype=np.uint8)
    for m in range(M):
        sub = samples[:, m * D_SUB:(m + 1) * D_SUB]            # (n_check, D_sub)
        # 거리: (n_check, K) — broadcast 로 계산
        dists = np.sum(
            (sub[:, None, :] - codebook[m][None, :, :]) ** 2,
            axis=2,
        )
        codes[:, m] = np.argmin(dists, axis=1)

    # 복원: 코드 → 평균 벡터
    reconstructed = np.zeros_like(samples)
    for m in range(M):
        reconstructed[:, m * D_SUB:(m + 1) * D_SUB] = codebook[m][codes[:, m]]

    # 코사인 유사도
    eps = 1e-8
    orig_norm = np.linalg.norm(samples, axis=1) + eps
    rec_norm  = np.linalg.norm(reconstructed, axis=1) + eps
    cos_sim = np.sum(samples * reconstructed, axis=1) / (orig_norm * rec_norm)

    logger.info(
        f"양자화 품질 검증 (n={n_check}) — "
        f"코사인 유사도 평균 {cos_sim.mean():.4f}, "
        f"min {cos_sim.min():.4f}, max {cos_sim.max():.4f}"
    )

    if cos_sim.mean() < 0.85:
        logger.warning(
            f"  ⚠ 평균 코사인 유사도가 0.85 미만 — 학습 샘플이 너무 적거나 "
            f"데이터 분포가 PQ 학습에 부적합할 수 있음."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PQ codebook 학습")
    parser.add_argument(
        "--src", type=Path, default=DEFAULT_SRC,
        help=f"학습용 이미지 디렉토리 (기본: {DEFAULT_SRC})",
    )
    parser.add_argument(
        "--max-samples", type=int, default=50_000,
        help="최대 학습 샘플 수 (기본: 50000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="샘플링 시드 (재현용)",
    )
    parser.add_argument(
        "--use-cache", action="store_true",
        help=f"{CACHE_PATH} 에 저장된 임베딩을 재사용 (이미지 처리 스킵)",
    )
    parser.add_argument(
        "--no-cache-save", action="store_true",
        help="추출한 임베딩을 캐시 파일로 저장하지 않음",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 임베딩 확보 ──
    if args.use_cache:
        if not CACHE_PATH.exists():
            raise FileNotFoundError(f"--use-cache 지정됨, 그러나 캐시 없음: {CACHE_PATH}")
        logger.info(f"캐시 로드: {CACHE_PATH}")
        embeddings = np.load(CACHE_PATH)
        logger.info(f"  → 임베딩 {embeddings.shape}")
    else:
        backbone = ResNet50Backbone(MODEL_DIR)
        if not backbone.is_loaded():
            raise RuntimeError(
                f"ResNet-50 ONNX 모델이 없음. workFile/export_resnet50.py 먼저 실행 필요."
            )

        image_paths = collect_image_paths(args.src, args.max_samples, args.seed)
        embeddings = extract_embeddings(image_paths, backbone)

        if not args.no_cache_save:
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            np.save(CACHE_PATH, embeddings)
            logger.info(f"임베딩 캐시 저장: {CACHE_PATH} ({embeddings.nbytes / 1024 / 1024:.1f}MB)")

    if len(embeddings) < K:
        raise ValueError(
            f"학습 샘플이 너무 적음: {len(embeddings)}개 < K={K}. "
            f"최소 수천 건 이상 필요."
        )

    # ── PQ 학습 ──
    codebook = train_pq_codebook(embeddings)

    # ── 검증 ──
    verify_quantization_quality(embeddings, codebook)

    # ── 저장 ──
    CODEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(CODEBOOK_PATH, codebook)
    size_mb = CODEBOOK_PATH.stat().st_size / 1024 / 1024
    logger.info(f"코드북 저장 완료: {CODEBOOK_PATH} ({size_mb:.2f}MB, shape {codebook.shape})")


if __name__ == "__main__":
    main()
