import cv2
import time
from ultralytics import YOLO

# Paths
MODEL_PATH = r"runs\detect\runs\ppe_baseline\weights\best.onnx"
VIDEO_PATH = r"videos\ppe_test.mp4"

# Load model
model = YOLO(MODEL_PATH)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Could not open video.")
    exit()

print("Starting inference... Press 'q' to quit.")

while True:
    # Read frame
    ret, frame = cap.read()

    if not ret:
        print("Video finished.")
        break

    # Preprocessing
    pre_start = time.perf_counter()

    input_frame = cv2.resize(frame, (640, 640))

    pre_end = time.perf_counter()

    # Inference
    infer_start = time.perf_counter()

    results = model.predict(
        source=input_frame,
        verbose=False
    )

    infer_end = time.perf_counter()

    # Postprocessing
    post_start = time.perf_counter()

    annotated_frame = results[0].plot()

    post_end = time.perf_counter()

    # Metrics
    pre_ms = (pre_end - pre_start) * 1000
    infer_ms = (infer_end - infer_start) * 1000
    post_ms = (post_end - post_start) * 1000

    total_ms = pre_ms + infer_ms + post_ms

    if total_ms > 0:
        fps = 1000 / total_ms
    else:
        fps = 0

    # Overlay metrics
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Pre: {pre_ms:.2f} ms",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Infer: {infer_ms:.2f} ms",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Post: {post_ms:.2f} ms",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "PPE Detection - ONNX FP16",
        annotated_frame
    )

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()