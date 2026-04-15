"""
백본 레지스트리 — import 시점에 모든 백본을 인스턴스화하고 REGISTRY 에 등록.

새 백본 추가:
  1. backbones/xxx.py 에 `XXXBackbone(Backbone)` 구현
  2. 아래 _BACKBONE_CLASSES 리스트에 클래스 추가
"""
from __future__ import annotations

import logging

from config import MODEL_DIR

from api.base import Backbone, BackboneOutput
from api.efficientnet import EfficientNetB0Backbone
from api.vgg16 import VGG16Backbone

logger = logging.getLogger(__name__)

_BACKBONE_CLASSES: list[type[Backbone]] = [
    VGG16Backbone,
    EfficientNetB0Backbone,
    # EfficientNetLite4Backbone,   # 향후 추가
]

REGISTRY: dict[str, Backbone] = {}
for _cls in _BACKBONE_CLASSES:
    _instance = _cls(MODEL_DIR)
    REGISTRY[_instance.name] = _instance
    _status = "loaded" if _instance.is_loaded() else "NOT loaded"
    logger.info(f"Backbone registered: {_instance.name} ({_status})")

__all__ = ["REGISTRY", "Backbone", "BackboneOutput"]
