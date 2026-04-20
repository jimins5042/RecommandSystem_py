"""EfficientNet-B0 백본 — 멀티-스테이지 가중 signature(704D) + 1280D 임베딩."""
from __future__ import annotations

import os

import numpy as np
import onnxruntime as ort
from PIL import Image

from api.base import (
    Backbone,
    BackboneOutput,
    mean_binarize_pack,
    to_float16_bytes,
    top_k_by_gap,
)

# ImageNet 정규화 상수 (NCHW)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class EfficientNetB0Backbone(Backbone):
    name = "efficientnet"
    display_name = "EfficientNet-B0"

    # Stage 5~8 project conv 출력 (build_feature_model.py 와 동기화 유지)
    # 각 스테이지 GAP → L2 정규화 → STAGE_WEIGHTS 곱 → concat = 80+112+192+320 = 704D signature
    STAGE_NODES = [
        '/features/features.4/features.4.2/block/block.3/block.3.0/Conv_output_0',
        '/features/features.5/features.5.2/block/block.3/block.3.0/Conv_output_0',
        '/features/features.6/features.6.3/block/block.3/block.3.0/Conv_output_0',
        '/features/features.7/features.7.0/block/block.3/block.3.0/Conv_output_0',
    ]
    # Stage 8 지배 + Stage 5~7 은 간헐 보조. STAGE_NODES 와 동일 순서.
    STAGE_WEIGHTS = [0.15, 0.15, 0.15, 1.0]
    EMBEDDING_NODE = '/Flatten_output_0'
    TOP_K = 40  # 704 채널 중 상위 K — 1차 필터링용

    ONNX_FILENAME = "efficientnet-b0-feat.onnx"

    def __init__(self, model_dir: str):
        path = os.path.join(model_dir, self.ONNX_FILENAME)
        self._session = ort.InferenceSession(path) if os.path.exists(path) else None

    def is_loaded(self) -> bool:
        return self._session is not None

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """ImageNet normalize, NCHW float32, [1,3,224,224]."""
        img = np.array(image.convert("RGB").resize(self.input_size), dtype=np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims((img - _MEAN) / _STD, axis=0).astype(np.float32)

    def extract(self, image: Image.Image) -> BackboneOutput:
        if not self.is_loaded():
            raise RuntimeError(f"{self.display_name} model not loaded. Run build_feature_model.py first.")

        arr = self._preprocess(image)
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(
            [*self.STAGE_NODES, self.EMBEDDING_NODE],
            {input_name: arr},
        )
        stage_feats = outputs[:-1]      # 4 개, each (1, C_i, H_i, W_i)
        embedding = outputs[-1][0]      # (1280,)

        # 각 stage feature map → channel-wise GAP → L2 정규화 → 가중치 곱 → concat (704D)
        # Stage 8 가중치 1.0 이 top-K 대부분을 차지해 semantic 변별력 유지,
        # Stage 5~7 은 0.15 로 축소되어 특출난 채널만 간헐 진입 (multi-scale 보조).
        gap_vecs = [s[0].mean(axis=(1, 2)) for s in stage_feats]
        signature = np.concatenate([
            w * (v / (np.linalg.norm(v) + 1e-8))
            for v, w in zip(gap_vecs, self.STAGE_WEIGHTS)
        ]).astype(np.float32)

        return BackboneOutput(
            order=top_k_by_gap(signature, channel_axis=0, k=self.TOP_K),
            features_bytes=mean_binarize_pack(embedding),
            embedding_bytes=to_float16_bytes(embedding),
        )
