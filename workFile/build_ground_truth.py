"""
Ground Truth 데이터 생성 스크립트.

같은 디렉토리의 images_*.csv(메타) + efficientNet_*.csv(임베딩)을 조인해서
카테고리별로 다양한 쿼리 이미지를 선정하고 FAISS 풀탐색으로 유사 후보를 추출한다.

실행:
    python workFile/build_ground_truth.py --csv-dir /path/to/csv

출력:
    ground_truth_data/queries.json    — 쿼리 + 후보 목록
    ground_truth_data/selections.json — 빈 선별 파일 (UI에서 채움)
"""
from __future__ import annotations

import argparse
import base64
import csv
import glob
import json
import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASS_NAMES, GT_DATA_DIR, KOREAN_TO_EN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

try:
    import faiss
    HAS_FAISS = True
    logger.info("FAISS 사용")
except ImportError:
    HAS_FAISS = False
    logger.warning("faiss-cpu 미설치 — numpy fallback 사용 (정확도 동일, 속도만 차이)")


def _url_to_english(image_url: str) -> str:
    """한글 경로를 영문으로 변환: /upload/가방/x.jpg → /upload/bag/x.jpg"""
    for ko, en in KOREAN_TO_EN.items():
        image_url = image_url.replace(ko, en)
    return image_url


def _category_from_url(image_url: str) -> str | None:
    """/upload/가방/x.jpg → 'bag'"""
    parts = image_url.strip("/").split("/")
    if len(parts) >= 2:
        return KOREAN_TO_EN.get(parts[1])
    return None


def load_image_meta(csv_dir: str) -> dict[str, dict]:
    """images_*.csv → {image_uuid: {image_url(영문), category}}"""
    pattern = os.path.join(csv_dir, "images_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"images_*.csv 없음: {csv_dir}")

    meta: dict[str, dict] = {}
    for csv_path in files:
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                url = row["image_url"]
                cat = _category_from_url(url)
                meta[row["image_uuid"]] = {
                    "image_url": _url_to_english(url),
                    "category": cat,
                }

    logger.info(f"이미지 메타 {len(meta)}건 로드 ({len(files)}개 파일)")
    return meta


def load_embeddings(csv_dir: str, meta: dict[str, dict]) -> dict[str, list[dict]]:
    """efficientNet_*.csv + 메타 조인 → 카테고리별 그룹."""
    pattern = os.path.join(csv_dir, "efficientNet_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"efficientNet_*.csv 없음: {csv_dir}")

    logger.info(f"임베딩 CSV {len(files)}개 로드...")
    groups: dict[str, list[dict]] = {name: [] for name in CLASS_NAMES}
    skipped = 0

    for csv_path in files:
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                uuid = row["image_uuid"]
                info = meta.get(uuid)
                if info is None or info["category"] not in groups:
                    skipped += 1
                    continue

                raw = base64.b64decode(row["img_embedding_base64"])
                embedding = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
                if embedding.shape[0] != 1280:
                    skipped += 1
                    continue

                groups[info["category"]].append({
                    "image_uuid": uuid,
                    "image_url": info["image_url"],
                    "embedding": embedding,
                })

    total = sum(len(v) for v in groups.values())
    logger.info(f"총 {total}건 로드 (스킵: {skipped})")
    for cat, items in groups.items():
        logger.info(f"  {cat}: {len(items)}건")
    return groups


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def search_topk_faiss(matrix: np.ndarray, query_idx: int, k: int) -> list[tuple[int, float]]:
    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    scores, indices = index.search(matrix[query_idx : query_idx + 1], k + 1)
    results = []
    for j in range(len(indices[0])):
        idx = int(indices[0][j])
        if idx == query_idx:
            continue
        results.append((idx, float(scores[0][j])))
        if len(results) >= k:
            break
    return results


def search_topk_numpy(matrix: np.ndarray, query_idx: int, k: int) -> list[tuple[int, float]]:
    scores = matrix @ matrix[query_idx]
    scores[query_idx] = -1.0
    top_indices = np.argsort(scores)[::-1][:k]
    return [(int(idx), float(scores[idx])) for idx in top_indices]


search_topk = search_topk_faiss if HAS_FAISS else search_topk_numpy


def select_diverse_queries(matrix: np.ndarray, n_queries: int, threshold: float) -> list[int]:
    """Greedy diverse selection: 쌍별 코사인 유사도 < threshold 보장."""
    n = matrix.shape[0]
    if n <= n_queries:
        return list(range(n))

    selected = [0]
    selected_vecs = matrix[0:1].copy()

    for _ in range(1, n_queries):
        sims = matrix @ selected_vecs.T
        max_sims = sims.max(axis=1)
        for idx in selected:
            max_sims[idx] = 2.0
        best = int(np.argmin(max_sims))
        if max_sims[best] >= threshold:
            logger.warning(
                f"  다양성 임계값({threshold}) 미달: "
                f"best_sim={max_sims[best]:.4f} (선택 {len(selected)+1}/{n_queries})"
            )
        selected.append(best)
        selected_vecs = np.vstack([selected_vecs, matrix[best : best + 1]])

    return selected


def process_category(
    cat: str,
    items: list[dict],
    n_queries: int,
    n_candidates: int,
    threshold: float,
) -> dict:
    logger.info(f"[{cat}] 처리 시작 ({len(items)}건)")
    embeddings = np.array([item["embedding"] for item in items], dtype=np.float32)
    embeddings = l2_normalize(embeddings)

    query_indices = select_diverse_queries(embeddings, n_queries, threshold)
    logger.info(f"[{cat}] 쿼리 {len(query_indices)}개 선정 완료")

    query_vecs = embeddings[query_indices]
    sim_matrix = query_vecs @ query_vecs.T
    np.fill_diagonal(sim_matrix, 0)
    logger.info(f"[{cat}] 쿼리 간 최대 유사도: {sim_matrix.max():.4f}")

    queries = []
    for i, qi in enumerate(query_indices):
        item = items[qi]
        results = search_topk(embeddings, qi, n_candidates)
        queries.append({
            "query_id": f"{cat}_{i:02d}",
            "image_uuid": item["image_uuid"],
            "image_url": item["image_url"],
            "candidates": [
                {
                    "image_uuid": items[idx]["image_uuid"],
                    "image_url": items[idx]["image_url"],
                    "similarity": round(sim, 6),
                }
                for idx, sim in results
            ],
        })

    logger.info(f"[{cat}] 완료")
    return {"total_images": len(items), "queries": queries}


def atomic_write_json(path: str, data: dict) -> None:
    dir_path = os.path.dirname(path)
    with tempfile.NamedTemporaryFile("w", dir=dir_path, suffix=".tmp", delete=False, encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        tmp_path = f.name
    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser(description="Ground Truth 쿼리 + 후보 생성")
    parser.add_argument("--csv-dir", required=True, help="images_*.csv + efficientNet_*.csv 디렉토리")
    parser.add_argument("--output-dir", default=GT_DATA_DIR, help="출력 디렉토리")
    parser.add_argument("--queries-per-cat", type=int, default=20, help="카테고리별 쿼리 수")
    parser.add_argument("--candidates-per-query", type=int, default=20, help="쿼리별 후보 수")
    parser.add_argument("--diversity-threshold", type=float, default=0.95, help="쿼리 간 최대 코사인 유사도")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    meta = load_image_meta(args.csv_dir)
    groups = load_embeddings(args.csv_dir, meta)

    categories = {}
    for cat in CLASS_NAMES:
        items = groups[cat]
        if not items:
            logger.warning(f"[{cat}] 데이터 없음 — 건너뜀")
            continue
        categories[cat] = process_category(
            cat, items, args.queries_per_cat, args.candidates_per_query, args.diversity_threshold
        )

    queries_data = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "csv_dir": os.path.abspath(args.csv_dir),
            "total_images": sum(len(groups[c]) for c in CLASS_NAMES),
            "diversity_threshold": args.diversity_threshold,
            "queries_per_category": args.queries_per_cat,
            "candidates_per_query": args.candidates_per_query,
        },
        "categories": categories,
    }

    queries_path = os.path.join(args.output_dir, "queries.json")
    atomic_write_json(queries_path, queries_data)
    logger.info(f"queries.json 저장: {queries_path}")

    selections_path = os.path.join(args.output_dir, "selections.json")
    if not os.path.exists(selections_path):
        atomic_write_json(selections_path, {})
        logger.info(f"selections.json 초기화: {selections_path}")
    else:
        logger.info(f"selections.json 이미 존재 — 덮어쓰지 않음")

    total_queries = sum(len(c["queries"]) for c in categories.values())
    logger.info(f"완료: {len(categories)}개 카테고리, {total_queries}개 쿼리")


if __name__ == "__main__":
    main()
