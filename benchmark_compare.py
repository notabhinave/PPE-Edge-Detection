import cv2
import time
from ultralytics import YOLO

VIDEO_PATH = r"videos\ppe_test.mp4"

MODELS = {
    "FP32_PT": r"runs\detect\runs\ppe_baseline\weights\best.pt",
    "ONNX_FP16": r"runs\detect\runs\ppe_baseline\weights\best.onnx"
}

NUM_FRAMES = 100

for model_name, model_path in MODELS.items():

    print(f"\nTesting {model_name}")

    model = YOLO(model_path)

    cap = cv2.VideoCapture(VIDEO_PATH)

    frame_count = 0
    total_time = 0

    while frame_count < NUM_FRAMES:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (640, 640))

        start = time.perf_counter()

        _ = model.predict(
            source=frame,
            verbose=False
        )

        end = time.perf_counter()

        total_time += (end - start)

        frame_count += 1

    cap.release()

    avg_ms = (total_time / frame_count) * 1000

    fps = 1000 / avg_ms

    print(f"Frames Tested : {frame_count}")
    print(f"Average Latency : {avg_ms:.2f} ms")
    print(f"Average FPS : {fps:.2f}")