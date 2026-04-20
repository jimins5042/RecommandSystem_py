"""
selections.json에서 카테고리별로 selected_image_ids가 많은 쿼리를 추려
테스트셋(test_queries.json)으로 저장.

실행:
    python workFile/rank_selections.py
    python workFile/rank_selections.py --top-n 15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASS_NAMES, GT_DATA_DIR

SELECTIONS_PATH = Path(GT_DATA_DIR) / "selections.json"
QUERIES_PATH    = Path(GT_DATA_DIR) / "queries.json"
OUTPUT_PATH     = Path(GT_DATA_DIR) / "test_queries.json"


def get_category(query_id: str) -> str | None:
    for cat in sorted(CLASS_NAMES, key=len, reverse=True):
        if query_id.startswith(cat + "_"):
            return cat
    return None


def build_query_meta(queries: dict) -> dict[str, dict]:
    """queries.json → {query_id: {image_uuid, image_url}}"""
    meta = {}
    for cat_data in queries["categories"].values():
        for q in cat_data["queries"]:
            meta[q["query_id"]] = {
                "image_uuid": q["image_uuid"],
                "image_url":  q["image_url"],
            }
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=10, help="카테고리별 선택할 쿼리 수 (기본 10)")
    args = parser.parse_args()

    with open(SELECTIONS_PATH, encoding="utf-8") as f:
        selections: dict = json.load(f)

    with open(QUERIES_PATH, encoding="utf-8") as f:
        query_meta = build_query_meta(json.load(f))

    # 카테고리별 그룹핑 + selected_image_ids 개수로 정렬
    groups: dict[str, list[dict]] = {cat: [] for cat in CLASS_NAMES}

    for query_id, data in selections.items():
        cat = get_category(query_id)
        if cat is None:
            continue
        meta = query_meta.get(query_id, {})
        groups[cat].append({
            "query_id":           query_id,
            "image_uuid":         meta.get("image_uuid", ""),
            "image_url":          meta.get("image_url", ""),
            "ground_truth_count": len(data["selected_image_ids"]),
            "ground_truth":       data["selected_image_ids"],
        })

    # 정렬 + 상위 N개 추출
    result = {}
    for cat in CLASS_NAMES:
        ranked = sorted(groups[cat], key=lambda x: x["ground_truth_count"], reverse=True)
        top = ranked[: args.top_n]
        result[cat] = top

        print(f"\n{'='*45}")
        print(f"  {cat}  (전체 {len(ranked)}개 쿼리 → 상위 {len(top)}개 선택)")
        print(f"{'='*45}")
        for i, q in enumerate(top):
            print(f"  {i+1:>2}. {q['query_id']}  정답 {q['ground_truth_count']}개  {q['image_url']}")
        if len(ranked) > args.top_n:
            cutoff = ranked[args.top_n - 1]["ground_truth_count"]
            print(f"  (컷오프: {cutoff}개 이상)")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
