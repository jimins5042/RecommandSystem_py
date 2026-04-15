"""
백본 추상 인터페이스 + 공통 유틸.

모든 백본은 동일한 출력 계약(BackboneOutput)을 지켜야 하며,
이를 통해 응답 스키마 / 검색 인덱스 / 비교 로직이 통일된다.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Optional

import numpy as np
from PIL import Image


@dataclass
class BackboneOutput:
    """백본 추론 결과의 공통 스키마."""
    order: str                           # JSON 문자열: top-K 채널 인덱스
    features_bytes: bytes                # 이진화 후 packbits 결과
    embedding_bytes: Optional[bytes] = None  # float16 raw embedding (cosine 재정렬용)


class Backbone(ABC):
    """이미지 특징점 추출 백본의 공통 인터페이스."""

    # 서브클래스가 반드시 정의
    name: ClassVar[str]           # URL 식별자 (예: "vgg16", "efficientnet-b0")
    display_name: ClassVar[str]   # 사람이 읽는 이름 (예: "VGG16")

    # 공통 하이퍼파라미터 (필요 시 override)
    input_size: ClassVar[tuple[int, int]] = (224, 224)

    @abstractmethod
    def is_loaded(self) -> bool:
        """ONNX 모델 파일이 존재하고 세션이 생성되었는지."""
        ...

    @abstractmethod
    def extract(self, image: Image.Image) -> BackboneOutput:
        """PIL 이미지를 받아 BackboneOutput 반환. 전처리/추론/후처리 포함."""
        ...


# ══════════════════════════════════════════════
# 공통 후처리 유틸
# ══════════════════════════════════════════════

def top_k_by_gap(feature_map: np.ndarray, *, channel_axis: int, k: int = 25) -> str:
    """
    Channel-wise GAP → 값이 큰 순으로 top-k 채널 인덱스.
    JSON 문자열로 반환 (프론트/DB 계약 유지).
    """
    other_axes = tuple(i for i in range(feature_map.ndim) if i != channel_axis)
    means = feature_map.mean(axis=other_axes)
    top_k = np.argsort(means)[::-1][:k].tolist()
    return json.dumps(top_k)


def mean_binarize_pack(features: np.ndarray) -> bytes:
    """
    0을 제외한 값들의 평균을 임계값으로 이진화 → packbits.
    Hamming 유사도 검색용 컴팩트 표현.
    """
    flat = features.flatten()
    nonzero = flat[flat != 0]
    mean_val = float(nonzero.mean()) if len(nonzero) > 0 else 0.0
    binary = np.where(flat >= mean_val, 1, 0).astype(np.uint8)
    return np.packbits(binary).tobytes()


def to_float16_bytes(features: np.ndarray) -> bytes:
    """float16 원본 임베딩 바이트열 — cosine 재정렬(re-rank) 단계용."""
    return features.astype(np.float16).tobytes()
