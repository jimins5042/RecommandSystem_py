"""
EfficientNet-B0 features.7 (7x7x320) 레이어 노드명 및 shape 확인
"""
import onnx
import onnxruntime as ort
import numpy as np
from onnx import helper, TensorProto
import os

MODEL_PATH = 'model/efficientNet-b0.onnx'

model = onnx.load(MODEL_PATH)

# features.7 관련 노드 전체 출력
print('=== features.7 노드 목록 ===')
for node in model.graph.node:
    outputs = list(node.output)
    if any('features.7' in o for o in outputs):
        print(f'  op={node.op_type:20s}  out={outputs}')

# features.7 마지막 Conv 출력 shape 확인
candidate = '/features/features.7/features.7.0/block/block.3/block.3.0/Conv_output_0'
print(f'\n=== 후보 노드 shape 확인 ===')
print(f'  {candidate}')

tmp_model = onnx.load(MODEL_PATH)
tmp_model.graph.output.append(
    helper.make_tensor_value_info(candidate, TensorProto.FLOAT, None)
)
onnx.save(tmp_model, 'model/_tmp_check.onnx')

sess = ort.InferenceSession('model/_tmp_check.onnx')
dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
result = sess.run([candidate], {'images': dummy})
print(f'  shape: {result[0].shape}')
os.remove('model/_tmp_check.onnx')
