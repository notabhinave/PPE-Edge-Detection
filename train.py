from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="construction-ppe/data.yaml",
    epochs=30,
    imgsz=640,
    batch=8,
    device="cpu",
    workers=2,
    project="runs",
    name="ppe_baseline"
)