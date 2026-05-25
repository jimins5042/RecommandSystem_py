"""
LabelMe JSON → YOLO 포맷 변환 + train/val 분할 스크립트

실행:
    python preprocess_data.py

출력 구조:
    yolo_dataset/
    ├── data.yaml
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path 에 추가 (스크립트 직접 실행 지원)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
import os
import shutil
import random

from config import CLASS_NAMES, OUTPUT_DIR, SOURCE_DIR

VAL_RATIO = 0.2  # 검증 데이터 비율
SEED = 42

# 파일명 접두사 [한글 카테고리] → class_id (CLASS_NAMES 순서와 매칭)
CLASS_MAP = {
    "가방":   CLASS_NAMES.index("bag"),
    "선글라스": CLASS_NAMES.index("sunglasses"),
    "식음료":  CLASS_NAMES.index("food_drink"),
    "신발":   CLASS_NAMES.index("shoes"),
    "의류":   CLASS_NAMES.index("clothing"),
}


def create_dirs():
    """YOLO 디렉토리 구조 생성"""
    for split in ("train", "val"):
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def extract_class_from_filename(filename: str) -> int | None:
    """파일명에서 [카테고리] 접두사를 파싱하여 class_id 반환"""
    if not filename.startswith("["):
        return None
    bracket_end = filename.find("]")
    if bracket_end == -1:
        return None
    category = filename[1:bracket_end].strip()
    return CLASS_MAP.get(category)


def labelme_to_yolo(json_path: Path) -> str | None:
    """LabelMe JSON → YOLO 라벨 텍스트 변환"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w = data.get("imageWidth", 0)
    img_h = data.get("imageHeight", 0)
    if img_w == 0 or img_h == 0:
        return None

    class_id = extract_class_from_filename(json_path.name)
    if class_id is None:
        return None

    lines = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue
        pts = shape["points"]
        x1, y1 = pts[0]
        x2, y2 = pts[1]

        # 좌표 정규화 (YOLO: center_x, center_y, width, height)
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w = abs(x2 - x1) / img_w
        h = abs(y2 - y1) / img_h

        # 0~1 범위로 클리핑
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))

        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return "\n".join(lines) if lines else None


def write_data_yaml():
    """YOLO data.yaml 생성"""
    yaml_content = f"""path: .
train: images/train
val: images/val

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    with open(OUTPUT_DIR / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)


def main():
    random.seed(SEED)
    create_dirs()

    # JSON 파일 목록 수집
    json_files = sorted(SOURCE_DIR.glob("*.json"))
    print(f"총 JSON 파일: {len(json_files)}개")

    # 카테고리별로 그룹핑 (stratified split을 위해)
    category_groups: dict[int, list[Path]] = {v: [] for v in CLASS_MAP.values()}
    skipped = 0

    for jf in json_files:
        class_id = extract_class_from_filename(jf.name)
        if class_id is not None:
            category_groups[class_id].append(jf)
        else:
            skipped += 1

    print(f"스킵된 파일: {skipped}개")
    for cat, cid in CLASS_MAP.items():
        print(f"  [{cat}] ({CLASS_NAMES[cid]}): {len(category_groups[cid])}장")

    # 카테고리별 stratified train/val 분할 + 변환
    stats = {"train": 0, "val": 0, "failed": 0}

    for class_id, files in category_groups.items():
        random.shuffle(files)
        val_count = int(len(files) * VAL_RATIO)
        val_files = set(f.stem for f in files[:val_count])

        for jf in files:
            split = "val" if jf.stem in val_files else "train"

            # YOLO 라벨 변환
            yolo_label = labelme_to_yolo(jf)
            if yolo_label is None:
                stats["failed"] += 1
                continue

            # 이미지 파일 찾기
            img_path = jf.with_suffix(".jpg")
            if not img_path.exists():
                img_path = jf.with_suffix(".png")
            if not img_path.exists():
                stats["failed"] += 1
                continue

            # 안전한 파일명 생성 (한글/특수문자 제거)
            safe_name = f"{CLASS_NAMES[class_id]}_{jf.stem.split('] ')[-1] if '] ' in jf.stem else jf.stem}"
            # 파일명에서 문제될 수 있는 문자 제거
            safe_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in safe_name)

            # 이미지 복사
            dst_img = OUTPUT_DIR / "images" / split / f"{safe_name}{img_path.suffix}"
            shutil.copy2(img_path, dst_img)

            # 라벨 저장
            dst_label = OUTPUT_DIR / "labels" / split / f"{safe_name}.txt"
            with open(dst_label, "w", encoding="utf-8") as f:
                f.write(yolo_label)

            stats[split] += 1

    write_data_yaml()

    print(f"\n변환 완료!")
    print(f"  Train: {stats['train']}장")
    print(f"  Val:   {stats['val']}장")
    print(f"  실패:  {stats['failed']}장")
    print(f"\n출력 경로: {OUTPUT_DIR}")
    print(f"data.yaml: {OUTPUT_DIR / 'data.yaml'}")


if __name__ == "__main__":
    main()
