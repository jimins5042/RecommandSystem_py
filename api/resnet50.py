"""ResNet-50 백본 — 단일 ONNX 세션, GAP 적용된 2048-dim 임베딩 출력.

ONNX 모델은 workFile/export_resnet50.py 로 생성:
  avgpool + flatten → (N, 2048)  (검색용 임베딩)
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import onnxruntime as ort
from PIL import Image

from api.base import (
    Backbone,
    BackboneOutput,
    mean_binarize_pack,
    pq_encode,
    to_float16_bytes,
    top_k_by_gap,
)

logger = logging.getLogger(__name__)

# ImageNet 정규화 상수 (NCHW) — torchvision V2 표준
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class ResNet50Backbone(Backbone):
    name = "resnet50"
    display_name = "ResNet-50"

    ONNX_FILENAME    = "resnet50.onnx"
    CODEBOOK_FILENAME = "pq_codebook.npy"

    def __init__(self, model_dir: str):
        onnx_path = os.path.join(model_dir, self.ONNX_FILENAME)
        self._session = ort.InferenceSession(onnx_path) if os.path.exists(onnx_path) else None

        # PQ 코드북 로드 (없으면 None — pq_code_bytes 가 None 으로 반환됨)
        cb_path = os.path.join(model_dir, self.CODEBOOK_FILENAME)
        if os.path.exists(cb_path):
            self._codebook: Optional[np.ndarray] = np.load(cb_path).astype(np.float32)
            logger.info(
                f"PQ codebook loaded from {cb_path}, shape={self._codebook.shape}"
            )
        else:
            self._codebook = None
            logger.info(
                f"PQ codebook not found at {cb_path}. "
                f"pq_code_bytes will be None. Run workFile/train_pq_codebook.py to enable."
            )

    def is_loaded(self) -> bool:
        return self._session is not None

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """ImageNet normalize, NCHW float32, [1,3,224,224]."""
        img = np.array(image.convert("RGB").resize(self.input_size), dtype=np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims((img - _MEAN) / _STD, axis=0).astype(np.float32)

    def extract(self, image: Image.Image) -> BackboneOutput:
        if not self.is_loaded():
            raise RuntimeError(
                f"{self.display_name} model not loaded. Run workFile/export_resnet50.py first."
            )

        arr = self._preprocess(image)
        input_name = self._session.get_inputs()[0].name
        # ONNX 출력: (1, 2048) — avgpool + flatten 까지 적용된 상태
        embedding = self._session.run(None, {input_name: arr})[0][0]  # (2048,)

        # 코드북이 있으면 PQ 인코딩, 없으면 None
        pq_bytes = pq_encode(embedding, self._codebook) if self._codebook is not None else None

        # GAP 출력 자체가 채널 평균값이므로 1D 그대로 top-K 적용 가능
        return BackboneOutput(
            order=top_k_by_gap(embedding, channel_axis=0, k=25),
            features_bytes=mean_binarize_pack(embedding),
            embedding_bytes=to_float16_bytes(embedding),
            pq_code_bytes=pq_bytes,
        )
