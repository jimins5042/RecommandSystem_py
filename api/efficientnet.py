"""EfficientNet-B0 백본 — single-session, 2개 중간 출력 노출."""
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

    # build_feature_model.py 에서 노출한 중간 출력 노드 이름
    STAGE1_NODE = "/features/features.7/features.7.0/block/block.3/block.3.0/Conv_output_0"
    STAGE2_NODE = "/Flatten_output_0"

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
        stage1, embedding = self._session.run(
            [self.STAGE1_NODE, self.STAGE2_NODE],
            {input_name: arr},
        )
        # stage1:    (1, 320, 7, 7)  → channel axis = 0 (after [0] squeeze)
        # embedding: (1, 1280)       → 0번째 샘플
        emb = embedding[0]

        return BackboneOutput(
            order=top_k_by_gap(stage1[0], channel_axis=0, k=25),
            features_bytes=mean_binarize_pack(emb),
            embedding_bytes=to_float16_bytes(emb),
        )
