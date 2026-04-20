"""
efficientNet-b0.onnx 에 멀티-스테이지 중간 출력과 1280D 임베딩을 노출.

구성:
  - STAGE_NODES  : Stage 5~8 project conv 출력 4개 (GAP 전 feature map)
                   80 + 112 + 192 + 320 = 704 채널
                   → 각각 GAP → L2 정규화 → 가중치(Stage 8=1.0, 나머지=0.15) → concat(704)
                   → 1차 필터링 signature
                   ※ Stage 8 지배 + Stage 5~7 은 간헐 보조 (semantic purity 우선)
  - EMBEDDING_NODE : 1280D Flatten 벡터
                   → 이진화(featuresBase64) + fp16(embedding) 재정렬용

실행:
    python workFile/build_feature_model.py
출력:
    model/efficientnet-b0-feat.onnx
"""
import onnx
from onnx import helper, TensorProto

SRC = 'model/efficientNet-b0.onnx'
DST = 'model/efficientnet-b0-feat.onnx'

# EfficientNet-B0 표준 구조 기준: (node_name, out_channels, last_block_idx_in_stage)
# features.2 (2 blocks) ~ features.7 (1 block) 의 마지막 블록 project conv 출력
STAGE_NODES: list[tuple[str, int]] = [
    ('/features/features.4/features.4.2/block/block.3/block.3.0/Conv_output_0',  80),
    ('/features/features.5/features.5.2/block/block.3/block.3.0/Conv_output_0', 112),
    ('/features/features.6/features.6.3/block/block.3/block.3.0/Conv_output_0', 192),
    ('/features/features.7/features.7.0/block/block.3/block.3.0/Conv_output_0', 320),
]
EMBEDDING_NODE = '/Flatten_output_0'

model = onnx.load(SRC)

# 기존 graph 의 노드 출력명 집합 — 존재 검증용
existing_outputs = {o for node in model.graph.node for o in node.output}
for name, _ in STAGE_NODES:
    if name not in existing_outputs:
        raise RuntimeError(f"노드 미발견: {name}\n→ inspect_model.py 로 실제 노드명 확인 필요")

# 4 개 stage 중간 출력 추가 (배치 dynamic)
for name, ch in STAGE_NODES:
    model.graph.output.append(
        helper.make_tensor_value_info(name, TensorProto.FLOAT, ['N', ch, None, None])
    )

# 1280D embedding 출력 추가
model.graph.output.append(
    helper.make_tensor_value_info(EMBEDDING_NODE, TensorProto.FLOAT, ['N', 1280])
)

# 입력 배치 dynamic
model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'N'
model.graph.input[0].type.tensor_type.shape.dim[0].ClearField('dim_value')

# 기존 1000D 출력도 dynamic
model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = 'N'
model.graph.output[0].type.tensor_type.shape.dim[0].ClearField('dim_value')

onnx.save(model, DST)
print(f"저장 완료: {DST}")
print(f"signature 차원: {sum(ch for _, ch in STAGE_NODES)} ({len(STAGE_NODES)} stages concat)")
print("출력 목록:")
for o in model.graph.output:
    print(f"  {o.name}")
