"""
torchvision ResNet-50(IMAGENET1K_V2) → ONNX export.

  - 출력: avgpool + flatten → (N, 2048)  (검색용 임베딩)
  - 배치 차원 dynamic
  - opset 17

실행:
    python workFile/export_resnet50.py
출력:
    model/resnet50.onnx
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import os

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from config import MODEL_DIR


class ResNet50Embedding(nn.Module):
    """ResNet-50 의 마지막 FC 직전 단계 (avgpool + flatten) 까지를 노출."""

    def __init__(self) -> None:
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # children: conv1, bn1, relu, maxpool, layer1~4, avgpool, fc
        # fc 직전까지 사용
        self.features = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)         # (N, 2048, 1, 1)
        return torch.flatten(x, 1)   # (N, 2048)


def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    out_path = os.path.join(MODEL_DIR, "resnet50.onnx")

    model = ResNet50Embedding().eval()
    dummy = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy,
        out_path,
        opset_version=17,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={
            "input":     {0: "N"},
            "embedding": {0: "N"},
        },
        # 신버전 PyTorch에서 dynamo 기반 exporter(onnxscript 의존)가 호출되는 것을 방지.
        # 기존 TorchScript 기반 exporter를 강제 사용.
        dynamo=False,
    )
    print(f"저장 완료: {out_path}")


if __name__ == "__main__":
    main()
