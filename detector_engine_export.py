import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from ultralytics import YOLO
import tensorrt as trt
import os
from config import BEST_WEIGHTS

# ================================================================
# PART 1: YOLO .pt → TensorRT Engine (direct, no manual ONNX step)
# ================================================================
# Ultralytics handles the full ONNX → TRT pipeline internally.
# Use fp16=True for speed without needing a calibration dataset.
# Use int8=True only if you have a data.yaml for calibration.

print("📦 Exporting YOLO model directly to TensorRT engine (TRT 10)...")

yolo_model = YOLO(BEST_WEIGHTS)

yolo_model.export(
    format="engine",    # Directly produces a .engine file
    imgsz=640,          # must match imgsz in predict()
    # dynamic=True,       # Variable input resolution not needed since imgsz is set
    batch=1,            # Max batch size (sets the optimization profile)
    workspace=8,        # GB of GPU workspace for the TRT builder
    half=True,          # FP16 precision — fast and no calibration needed
    # int8=True,        # Uncomment + add data="data.yaml" if you want INT8
    # data="data.yaml", # Required only for INT8 calibration
)
                
print("✅ YOLO TensorRT engine saved as best.engine\n")

# Quick inference test to verify the engine works
yolo_engine = YOLO(
    str(BEST_WEIGHTS).replace(".pt", ".engine"),
    task="detect"
)
print("🔍 Running test inference on YOLO engine...")
test_img = "./dataset/trt_test/knife.jpg"
results = yolo_engine.predict(source=test_img, imgsz=640, conf=0.25, verbose=False)
print(f"✅ YOLO engine inference OK — {len(results)} result(s) returned\n")
      