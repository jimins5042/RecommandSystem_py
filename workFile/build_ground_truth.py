"""
Ground Truth 데이터 생성 스크립트.

카테고리별 다양한 쿼리 이미지를 선정하고,
FAISS 풀탐색으로 각 쿼리의 유사 이미지 후보를 추출한다.

실행:
    python workFile/build_ground_truth.py --csv /path/to/embeddings.csv

출력:
    ground_truth_data/queries.json   — 쿼리 + 후보 목록
    ground_truth_data/selections.json — 빈 선별 파일 (UI에서 채움)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASS_NAMES, GT_DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# FAISS import (없으면 numpy fallback)
try:
    import faiss

    HAS_FAISS = True
    logger.info("FAISS 사용")
except ImportError:
    HAS_FAISS = False
    logger.warning("faiss-cpu 미설치 — numpy fallback 사용 (정확도 동일, 속도만 차이)")


def load_csv(csv_path: str) -> dict[str, list[dict]]:
    """CSV 로드 → 카테고리별 그룹핑.

    Returns:
        {category: [{image_id, image_url, embedding: np.array(1280,)}]}
    """
    logger.info(f"CSV 로드: {csv_path}")
    groups: dict[str, list[dict]] = {name: [] for name in CLASS_NAMES}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        # dim 컬럼 이름 추출
        dim_cols = [c for c in fieldnames if c.startswith("dim_")]
        if len(dim_cols) != 1280:
            raise ValueError(f"dim_ 컬럼이 {len(dim_cols)}개 (1280 필요). 컬럼명 확인 필요.")

        count = 0
        for row in reader:
            cat = row["category"]
            if cat not in groups:
                logger.warning(f"알 수 없는 카테고리 '{cat}' — 건너뜀")
                continue

            embedding = np.array([float(row[c]) for c in dim_cols], dtype=np.float32)
            groups[cat].append({
                "image_id": row["image_id"],
                "image_url": row["image_url"],
                "embedding": embedding,
            })
            count += 1

    logger.info(f"총 {count}건 로드")
    for cat, items in groups.items():
        logger.info(f"  {cat}: {len(items)}건")

    return groups


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """L2 정규화 (코사인 유사도용)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def search_topk_faiss(matrix: np.ndarray, query_idx: int, k: int) -> list[tuple[int, float]]:
    """FAISS IndexFlatIP로 top-k 검색 (자기 자신 제외)."""
    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    query_vec = matrix[query_idx : query_idx + 1]
    scores, indices = index.search(query_vec, k + 1)

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
    """numpy fallback: 내적 → argsort."""
    query_vec = matrix[query_idx]
    scores = matrix @ query_vec  # (N,)
    scores[query_idx] = -1.0  # 자기 자신 제외

    top_indices = np.argsort(scores)[::-1][:k]
    return [(int(idx), float(scores[idx])) for idx in top_indices]


search_topk = search_topk_faiss if HAS_FAISS else search_topk_numpy


def select_diverse_queries(
    matrix: np.ndarray,
    n_queries: int,
    threshold: float,
) -> list[int]:
    """Greedy diverse selection: 쌍별 코사인 유사도 < threshold 보장.

    가장 다양한 쿼리를 선정하기 위해, 기존 선택과의 최대 유사도가
    가장 낮은 후보를 반복적으로 선택한다.
    """
    n = matrix.shape[0]
    if n <= n_queries:
        return list(range(n))

    selected = [0]
    selected_vecs = matrix[0:1].copy()  # (1, dim)

    for _ in range(1, n_queries):
        # 모든 후보와 기존 선택 간 유사도 계산
        sims = matrix @ selected_vecs.T  # (N, len(selected))
        max_sims = sims.max(axis=1)  # (N,)

        # 이미 선택된 인덱스는 제외
        for idx in selected:
            max_sims[idx] = 2.0

        # 최대 유사도가 가장 낮은 후보 선택 (가장 다양한)
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
    """한 카테고리 처리: 쿼리 선정 + 후보 검색."""
    logger.info(f"[{cat}] 처리 시작 ({len(items)}건)")

    # 임베딩 행렬 구성 + L2 정규화
    embeddings = np.array([item["embedding"] for item in items], dtype=np.float32)
    embeddings = l2_normalize(embeddings)

    # 쿼리 선정
    logger.info(f"[{cat}] 다양한 쿼리 {n_queries}개 선정 중...")
    query_indices = select_diverse_queries(embeddings, n_queries, threshold)
    logger.info(f"[{cat}] 쿼리 {len(query_indices)}개 선정 완료")

    # 선정된 쿼리 간 유사도 검증
    query_vecs = embeddings[query_indices]
    sim_matrix = query_vecs @ query_vecs.T
    np.fill_diagonal(sim_matrix, 0)
    max_pair_sim = sim_matrix.max()
    logger.info(f"[{cat}] 쿼리 간 최대 유사도: {max_pair_sim:.4f} (임계값: {threshold})")

    # 각 쿼리별 후보 검색
    queries = []
    for i, qi in enumerate(query_indices):
        query_id = f"{cat}_{i:02d}"
        item = items[qi]

        results = search_topk(embeddings, qi, n_candidates)

        candidates = []
        for idx, sim in results:
            candidates.append({
                "image_id": items[idx]["image_id"],
                "image_url": items[idx]["image_url"],
                "similarity": round(sim, 6),
            })

        queries.append({
            "query_id": query_id,
            "image_id": item["image_id"],
            "image_url": item["image_url"],
            "candidates": candidates,
        })

        logger.info(
            f"[{cat}] 쿼리 {query_id}: "
            f"top-1 sim={results[0][1]:.4f}, top-{n_candidates} sim={results[-1][1]:.4f}"
        )

    return {
        "total_images": len(items),
        "queries": queries,
    }


def atomic_write_json(path: str, data: dict) -> None:
    """원자적 JSON 파일 쓰기 (Windows 호환)."""
    dir_path = os.path.dirname(path)
    with tempfile.NamedTemporaryFile("w", dir=dir_path, suffix=".tmp", delete=False, encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        tmp_path = f.name
    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser(description="Ground Truth 쿼리 + 후보 생성")
    parser.add_argument("--csv", required=True, help="임베딩 CSV 파일 경로")
    parser.add_argument("--output-dir", default=GT_DATA_DIR, help="출력 디렉토리")
    parser.add_argument("--queries-per-cat", type=int, default=20, help="카테고리별 쿼리 수")
    parser.add_argument("--candidates-per-query", type=int, default=20, help="쿼리별 후보 수")
    parser.add_argument("--diversity-threshold", type=float, default=0.95, help="쿼리 간 최대 코사인 유사도")
    args = parser.parse_args()

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # CSV 로드
    groups = load_csv(args.csv)

    # 카테고리별 처리
    categories = {}
    for cat in CLASS_NAMES:
        items = groups[cat]
        if len(items) == 0:
            logger.warning(f"[{cat}] 데이터 없음 — 건너뜀")
            continue

        categories[cat] = process_category(
            cat, items, args.queries_per_cat, args.candidates_per_query, args.diversity_threshold
        )

    # queries.json 저장
    queries_data = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "csv_path": os.path.abspath(args.csv),
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

    # selections.json 초기화 (기존 파일이 없을 때만)
    selections_path = os.path.join(args.output_dir, "selections.json")
    if not os.path.exists(selections_path):
        atomic_write_json(selections_path, {})
        logger.info(f"selections.json 초기화: {selections_path}")
    else:
        logger.info(f"selections.json 이미 존재 — 덮어쓰지 않음")

    # 요약
    total_queries = sum(len(c["queries"]) for c in categories.values())
    logger.info(f"완료: {len(categories)}개 카테고리, {total_queries}개 쿼리")


if __name__ == "__main__":
    main()
