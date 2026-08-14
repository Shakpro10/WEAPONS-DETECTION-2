import json
import numpy as np
import torch
import torch.nn as nn
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from datetime import datetime
from torchvision.models import mobilenet_v3_small
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os


# ================================================================
# CONFIG (ARGUMENTS)
# ================================================================
CLASSIFIER_PTH    = "classifier_weights/best_model.pth"
CLASSIFIER_ENGINE = "classifier_weights/classifier_fp16.engine"
TEST_DIR          = "dataset/classifier_dataset/test"
JSON_OUTPUT       = "verification_reports/classifier_verification.json"

NUM_CLASSES       = 6
BATCH_SIZE_PTH    = 32
BATCH_SIZE_TRT    = 1

INPUT_SHAPE       = (1, 3, 224, 224)


# ================================================================
# TRANSFORM
# ================================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ================================================================
# UTIL
# ================================================================
def save_json(data: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"💾 Metrics saved to {path}")


def load_test_dataset():
    dataset = ImageFolder(root=TEST_DIR, transform=transform)

    pth_loader = DataLoader(dataset, batch_size=BATCH_SIZE_PTH, shuffle=False)
    trt_loader = DataLoader(dataset, batch_size=BATCH_SIZE_TRT, shuffle=False)

    print(f"✅ Dataset: {len(dataset)} images | {len(dataset.classes)} classes")
    print(f"   Classes: {dataset.classes}")

    return pth_loader, trt_loader, dataset.classes


# ================================================================
# PTH EVALUATION
# ================================================================
def evaluate_pth(dataloader):
    print("\n" + "=" * 60)
    print("BASELINE — PyTorch FP32")
    print("=" * 60)

    model = mobilenet_v3_small()
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        NUM_CLASSES
    )

    model.load_state_dict(
        torch.load(CLASSIFIER_PTH, map_location="cuda", weights_only=True)
    )

    model.eval().cuda()

    correct, total = 0, 0
    class_correct, class_total = {}, {}

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.cuda(), labels.cuda()

            outputs = model(images)
            preds   = torch.argmax(outputs, dim=1)

            total   += labels.size(0)
            correct += (preds == labels).sum().item()

            for l, p in zip(labels, preds):
                cls = l.item()
                class_correct[cls] = class_correct.get(cls, 0) + int(p == l)
                class_total[cls]   = class_total.get(cls, 0) + 1

    acc = correct / total
    print(f"   Accuracy: {acc:.4f} ({correct}/{total})")

    return acc, class_correct, class_total


# ================================================================
# TRT EVALUATION (FIXED FOR TRT 10)
# ================================================================
def evaluate_engine(dataloader):
    print("\n" + "=" * 60)
    print("TRT ENGINE — FP16 (TensorRT 10)")
    print("=" * 60)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    with open(CLASSIFIER_ENGINE, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # ---- Get tensor names (VERY IMPORTANT for TRT 10)
    input_name  = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    print(f"🔹 Input tensor : {input_name}")
    print(f"🔹 Output tensor: {output_name}")

    # ---- Allocate GPU memory
    d_input  = cuda.mem_alloc(int(np.prod(INPUT_SHAPE)) * np.float32().itemsize)
    d_output = cuda.mem_alloc(NUM_CLASSES * np.float32().itemsize)

    stream = cuda.Stream()

    correct, total = 0, 0
    class_correct, class_total = {}, {}

    for images, labels in dataloader:

        images_np = images.detach().cpu().numpy().astype(np.float32)

        # Copy input to GPU
        cuda.memcpy_htod_async(d_input, images_np.ravel(), stream)

        # ================================
        # TRT 10 EXECUTION API
        # ================================
        context.set_tensor_address(input_name, int(d_input))
        context.set_tensor_address(output_name, int(d_output))

        context.execute_async_v3(stream_handle=stream.handle)

        # Copy output back
        output_np = np.empty((1, NUM_CLASSES), dtype=np.float32)
        cuda.memcpy_dtoh_async(output_np, d_output, stream)
        stream.synchronize()

        pred = np.argmax(output_np, axis=1)[0]
        label = labels.numpy()[0]

        total += 1
        correct += int(pred == label)

        class_correct[label] = class_correct.get(label, 0) + int(pred == label)
        class_total[label]   = class_total.get(label, 0) + 1

    acc = correct / total
    print(f"   Accuracy: {acc:.4f} ({correct}/{total})")

    return acc, class_correct, class_total


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":

    pth_loader, trt_loader, class_names = load_test_dataset()

    pth_acc, pth_cc, pth_ct = evaluate_pth(pth_loader)
    engine_acc, eng_cc, eng_ct = evaluate_engine(trt_loader)

    # ---- per class report
    report = {}

    for idx, name in enumerate(class_names):
        pth_cls = pth_cc.get(idx, 0) / pth_ct.get(idx, 1)
        trt_cls = eng_cc.get(idx, 0) / eng_ct.get(idx, 1)

        report[name] = {
            "pth_fp32_accuracy": round(float(pth_cls), 4),
            "trt_fp16_accuracy": round(float(trt_cls), 4),
            "drop": round(float(pth_cls - trt_cls), 4),
        }

    overall_drop = pth_acc - engine_acc

    if overall_drop <= 0.01:
        verdict = "✅ Safe for production"
    elif overall_drop <= 0.02:
        verdict = "⚠️ Acceptable but monitor"
    else:
        verdict = "❌ Too much drop"

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"PTH Accuracy  : {pth_acc:.4f}")
    print(f"TRT Accuracy  : {engine_acc:.4f}")
    print(f"Drop          : {overall_drop:+.4f}")
    print(f"Verdict       : {verdict}")

    final_report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pth_acc": float(pth_acc),
        "trt_acc": float(engine_acc),
        "drop": float(overall_drop),
        "verdict": verdict,
        "per_class": report
    }

    save_json(final_report, JSON_OUTPUT)