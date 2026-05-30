from ultralytics import YOLO

model = YOLO(
    r"runs\detect\runs\ppe_baseline\weights\best.pt"
)

model.export(
    format="onnx",
    half=True,
    simplify=True
)