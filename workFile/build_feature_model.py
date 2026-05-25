"""
efficientNet-b0.onnx 에 두 개의 중간 레이어를 추가 output으로 노출하고,
배치 차원을 dynamic으로 변경합니다.

  - STAGE1_NODE : 7x7x320 feature map  → GAP → Top-K 인덱스 (order)
  - STAGE2_NODE : 1280D Flatten 벡터   → 이진화 (featuresBase64)

실행:
    python build_feature_model.py
출력:
    model/efficientnet-b0-feat.onnx
"""
import onnx
from onnx import helper, TensorProto

SRC = 'model/efficientNet-b0.onnx'
DST = 'model/efficientnet-b0-feat.onnx'

STAGE1_NODE = '/features/features.7/features.7.0/block/block.3/block.3.0/Conv_output_0'
STAGE2_NODE = '/Flatten_output_0'

model = onnx.load(SRC)

# 중간 레이어 output 추가 (배치 차원 dynamic)
model.graph.output.append(
    helper.make_tensor_value_info(STAGE1_NODE, TensorProto.FLOAT, ['N', 320, 7, 7])
)
model.graph.output.append(
    helper.make_tensor_value_info(STAGE2_NODE, TensorProto.FLOAT, ['N', 1280])
)

# 입력 배치 차원 dynamic 설정
model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'N'
model.graph.input[0].type.tensor_type.shape.dim[0].ClearField('dim_value')

# 기존 출력(output 1000D)도 dynamic
model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = 'N'
model.graph.output[0].type.tensor_type.shape.dim[0].ClearField('dim_value')

onnx.save(model, DST)
print(f"저장 완료: {DST}")
print("출력 목록:")
for o in model.graph.output:
    print(f"  {o.name}")
