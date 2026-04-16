"""
Ground Truth 선별 UI용 API 엔드포인트.

  GET  /gt/status                  — 카테고리별 진행률
  GET  /gt/queries                 — 전체 쿼리 목록 (완료 여부 포함)
  GET  /gt/query/{query_id}        — 쿼리 상세 + 후보 + 기존 선택
  POST /gt/query/{query_id}/select — 선택한 10건 저장
  GET  /gt/export                  — 최종 ground truth JSON 다운로드

사전 조건: workFile/build_ground_truth.py 로 queries.json 생성 필요.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import GT_DATA_DIR

router = APIRouter(tags=["ground-truth"])
logger = logging.getLogger(__name__)

QUERIES_PATH = os.path.join(GT_DATA_DIR, "queries.json")
SELECTIONS_PATH = os.path.join(GT_DATA_DIR, "selections.json")
EXPORT_PATH = os.path.join(GT_DATA_DIR, "ground_truth_final.json")

# queries.json 캐시 (불변 데이터 — 한 번만 로드)
_queries_cache: dict | None = None


def _load_queries() -> dict:
    """queries.json 로드 (캐시)."""
    global _queries_cache
    if _queries_cache is not None:
        return _queries_cache

    if not os.path.exists(QUERIES_PATH):
        raise HTTPException(
            404,
            "queries.json이 없습니다. 먼저 build_ground_truth.py를 실행하세요.",
        )

    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        _queries_cache = json.load(f)
    return _queries_cache


def _load_selections() -> dict:
    """selections.json 로드 (매번 읽기 — 작은 파일)."""
    if not os.path.exists(SELECTIONS_PATH):
        return {}
    with open(SELECTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_selections(data: dict) -> None:
    """selections.json 원자적 저장."""
    os.makedirs(GT_DATA_DIR, exist_ok=True)
    dir_path = os.path.dirname(SELECTIONS_PATH)
    with tempfile.NamedTemporaryFile(
        "w", dir=dir_path, suffix=".tmp", delete=False, encoding="utf-8"
    ) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        tmp_path = f.name
    os.replace(tmp_path, SELECTIONS_PATH)


def _find_query(query_id: str) -> tuple[str, dict]:
    """query_id로 카테고리 + 쿼리 데이터 조회."""
    queries = _load_queries()
    for cat, cat_data in queries["categories"].items():
        for q in cat_data["queries"]:
            if q["query_id"] == query_id:
                return cat, q
    raise HTTPException(404, f"쿼리 '{query_id}'를 찾을 수 없습니다.")


@router.get("/status")
async def get_status():
    """카테고리별 진행률."""
    queries = _load_queries()
    selections = _load_selections()

    categories = {}
    overall_total = 0
    overall_completed = 0

    for cat, cat_data in queries["categories"].items():
        total = len(cat_data["queries"])
        completed = sum(
            1 for q in cat_data["queries"] if q["query_id"] in selections
        )
        categories[cat] = {"total": total, "completed": completed}
        overall_total += total
        overall_completed += completed

    return {
        "categories": categories,
        "overall": {"total": overall_total, "completed": overall_completed},
    }


@router.get("/queries")
async def list_queries():
    """전체 쿼리 목록 (카테고리별, 완료 여부 포함)."""
    queries = _load_queries()
    selections = _load_selections()

    result = {}
    for cat, cat_data in queries["categories"].items():
        result[cat] = [
            {
                "query_id": q["query_id"],
                "image_id": q["image_id"],
                "image_url": q["image_url"],
                "completed": q["query_id"] in selections,
            }
            for q in cat_data["queries"]
        ]

    return {"categories": result}


@router.get("/query/{query_id}")
async def get_query(query_id: str):
    """쿼리 상세 + 후보 20건 + 기존 선택."""
    cat, query = _find_query(query_id)
    selections = _load_selections()

    selected = selections.get(query_id, {}).get("selected_image_ids", None)

    return {
        "query_id": query["query_id"],
        "category": cat,
        "image_id": query["image_id"],
        "image_url": query["image_url"],
        "candidates": query["candidates"],
        "selected_image_ids": selected,
    }


class SelectionRequest(BaseModel):
    selected_image_ids: list[str]


@router.post("/query/{query_id}/select")
async def save_selection(query_id: str, body: SelectionRequest):
    """선택한 10건 저장."""
    _, query = _find_query(query_id)

    # 검증: 정확히 10건
    if len(body.selected_image_ids) != 10:
        raise HTTPException(
            422,
            f"정확히 10건을 선택해야 합니다 (현재: {len(body.selected_image_ids)}건).",
        )

    # 검증: 모두 후보 목록에 포함
    candidate_ids = {c["image_id"] for c in query["candidates"]}
    invalid = set(body.selected_image_ids) - candidate_ids
    if invalid:
        raise HTTPException(
            422, f"후보 목록에 없는 이미지: {invalid}"
        )

    # 검증: 중복 없음
    if len(set(body.selected_image_ids)) != 10:
        raise HTTPException(422, "중복된 이미지 ID가 있습니다.")

    # 저장
    selections = _load_selections()
    selections[query_id] = {
        "selected_image_ids": body.selected_image_ids,
        "completed_at": datetime.now().isoformat(),
    }
    _save_selections(selections)

    logger.info(f"[ground_truth] {query_id} 선택 저장 완료")
    return {"ok": True}


@router.get("/export")
async def export_ground_truth():
    """최종 ground truth JSON 조립 + 다운로드."""
    queries = _load_queries()
    selections = _load_selections()

    ground_truth = []
    for cat, cat_data in queries["categories"].items():
        for q in cat_data["queries"]:
            sel = selections.get(q["query_id"])
            if sel is None:
                continue

            selected_ids = set(sel["selected_image_ids"])
            relevant = [
                c for c in q["candidates"] if c["image_id"] in selected_ids
            ]

            ground_truth.append({
                "query_id": q["query_id"],
                "category": cat,
                "query_image_id": q["image_id"],
                "query_image_url": q["image_url"],
                "relevant_images": relevant,
            })

    result = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_queries": len(ground_truth),
            "selections_per_query": 10,
        },
        "ground_truth": ground_truth,
    }

    # 파일로도 저장
    _save_selections_path = EXPORT_PATH
    os.makedirs(GT_DATA_DIR, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=GT_DATA_DIR, suffix=".tmp", delete=False, encoding="utf-8"
    ) as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        tmp_path = f.name
    os.replace(tmp_path, _save_selections_path)

    logger.info(f"[ground_truth] export 완료: {len(ground_truth)}건")
    return JSONResponse(
        content=result,
        headers={"Content-Disposition": "attachment; filename=ground_truth_final.json"},
    )
