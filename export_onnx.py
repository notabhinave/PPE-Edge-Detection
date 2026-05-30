from ultralytics import YOLO

model = YOLO("runs/detect/runs/ppe_baseline/weights/best.pt")

model.export(
    format="onnx",
    half=True
)