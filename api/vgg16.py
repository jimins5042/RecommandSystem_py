"""VGG16 백본 — 2-세션 구성 (full + block5_conv3)."""
from __future__ import annotations

import os

import numpy as np
import onnxruntime as ort
from PIL import Image

from api.base import (
    Backbone,
    BackboneOutput,
    mean_binarize_pack,
    top_k_by_gap,
)


class VGG16Backbone(Backbone):
    name = "vgg16"
    display_name = "VGG16"

    def __init__(self, model_dir: str):
        full_path   = os.path.join(model_dir, "vgg16_full.onnx")
        block5_path = os.path.join(model_dir, "vgg16_block5_conv3.onnx")
        if os.path.exists(full_path) and os.path.exists(block5_path):
            self._session_full   = ort.InferenceSession(full_path)
            self._session_block5 = ort.InferenceSession(block5_path)
        else:
            self._session_full = None
            self._session_block5 = None

    def is_loaded(self) -> bool:
        return self._session_full is not None and self._session_block5 is not None

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """RGB→BGR + ImageNet 채널별 평균 차감 (TF Caffe 모드), NHWC float32."""
        img = image.convert("RGB").resize(self.input_size)
        arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
        arr = arr[..., ::-1].copy()  # RGB → BGR
        arr[..., 0] -= 103.939
        arr[..., 1] -= 116.779
        arr[..., 2] -= 123.68
        return arr

    def extract(self, image: Image.Image) -> BackboneOutput:
        if not self.is_loaded():
            raise RuntimeError(f"{self.display_name} model not loaded")

        arr = self._preprocess(image)

        # full → 1D vector → mean binarize
        full_input = self._session_full.get_inputs()[0].name
        features = self._session_full.run(None, {full_input: arr})[0].flatten()

        # block5 → (1, 14, 14, 512) → squeeze → (14, 14, 512)
        block5_input = self._session_block5.get_inputs()[0].name
        fmap = self._session_block5.run(None, {block5_input: arr})[0].squeeze()

        return BackboneOutput(
            order=top_k_by_gap(fmap, channel_axis=2, k=25),
            features_bytes=mean_binarize_pack(features),
        )
