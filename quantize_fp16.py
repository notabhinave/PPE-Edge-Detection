import onnx
from onnx import TensorProto

model = onnx.load(
    r"runs\detect\runs\ppe_baseline\weights\best.onnx"
)

for initializer in model.graph.initializer:
    print(initializer.data_type)
    break