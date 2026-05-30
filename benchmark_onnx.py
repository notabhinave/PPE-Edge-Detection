from ultralytics import YOLO
import time

model = YOLO("runs/detect/runs/ppe_baseline/weights/best.onnx")

start = time.time()

metrics = model.val(
    data="construction-ppe/data.yaml"
)

end = time.time()

print(metrics)
print("Validation Time:", end - start)