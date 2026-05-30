# Real-Time Construction PPE Compliance Detection using YOLOv8n and ONNX FP16

## Overview

This project implements a real-time Personal Protective Equipment (PPE) compliance detection system for construction and industrial environments. The system detects workers and PPE items such as helmets, gloves, vests, boots, and goggles while also identifying safety violations such as missing PPE.

The trained YOLOv8n model was optimized for edge deployment by exporting it to ONNX with FP16 precision. The final system performs real-time inference on video streams while displaying detection results, confidence scores, FPS, and latency metrics.

---

## Problem Statement

Construction sites and industrial workplaces require workers to wear proper Personal Protective Equipment (PPE) to ensure safety and comply with regulations.

Manual monitoring is inefficient and prone to human error. This project automates PPE compliance monitoring using computer vision and object detection techniques.

### Objectives

* Detect workers and PPE items in real time.
* Identify missing PPE violations.
* Optimize the model for edge deployment.
* Measure real-time inference performance.

---

## Dataset

### Construction PPE Dataset

* Total Images: 1416
* Training Images: 1132
* Validation Images: 143
* Test Images: 141
* Format: YOLO

### Classes

| ID | Class     |
| -- | --------- |
| 0  | helmet    |
| 1  | gloves    |
| 2  | vest      |
| 3  | boots     |
| 4  | goggles   |
| 5  | none      |
| 6  | Person    |
| 7  | no_helmet |
| 8  | no_goggle |
| 9  | no_gloves |
| 10 | no_boots  |

---

## Model Architecture

### Baseline Model

* Model: YOLOv8n
* Framework: Ultralytics YOLO
* Precision: FP32
* Input Size: 640 × 640
* Training Epochs: 30

### Edge Optimized Model

* Format: ONNX
* Precision: FP16
* Runtime: ONNX Runtime
* Deployment Target: Edge Devices

---

## Training

Training was performed using the YOLOv8n architecture on the Construction PPE dataset.

Example training command:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="construction-ppe/data.yaml",
    epochs=30,
    imgsz=640,
    batch=8
)
```

---

## Model Conversion

The trained FP32 model was exported to ONNX FP16 format for efficient edge deployment.

```python
from ultralytics import YOLO

model = YOLO("best.pt")

model.export(
    format="onnx",
    half=True
)
```

---

## Performance Benchmark

### Accuracy Comparison

| Metric    | FP32 YOLOv8n | ONNX FP16 |
| --------- | ------------ | --------- |
| Precision | 51.77%       | 57.45%    |
| Recall    | 54.84%       | 55.99%    |
| mAP50     | 54.98%       | 57.04%    |
| mAP50-95  | 27.01%       | 28.16%    |

### Edge Deployment Comparison

| Metric          | FP32 YOLOv8n | ONNX FP16 |
| --------------- | ------------ | --------- |
| Model Size      | 6.25 MB      | 6.17 MB   |
| Average Latency | 48.25 ms     | 31.09 ms  |
| Average FPS     | 20.73        | 32.16     |

### Observations

* ONNX FP16 reduced inference latency by approximately 35%.
* FPS improved from 20.73 to 32.16.
* Model size decreased slightly after conversion.
* Detection accuracy remained comparable after optimization.

---

## Live Inference Features

The deployment script provides:

* Real-time PPE detection
* Bounding boxes
* Class labels
* Confidence scores
* FPS monitoring
* Pre-processing latency
* Inference latency
* Post-processing latency

Run:

```bash
python live_inference.py
```

---

## Project Structure

```text
PPE-Edge-Detection/
│
├── construction-ppe/
├── videos/
├── runs/
│
├── train.py
├── export_onnx.py
├── benchmark_compare.py
├── live_inference.py
├── requirements.txt
└── README.md
```

---

## Model Weights

### FP32 Model/ONNX FP16 Model

(https://drive.google.com/drive/folders/1SiDx-QXQY4p1lhetiz1FlUBK-_5cafX8?usp=drive_link)


## Demo Video

Add your YouTube / Google Drive / Loom video link here.

---

## Conclusion

This project demonstrates the deployment of an edge-optimized PPE compliance monitoring system using YOLOv8n and ONNX FP16. The optimized model achieved real-time performance of over 32 FPS while maintaining comparable detection accuracy, making it suitable for industrial safety monitoring applications.
