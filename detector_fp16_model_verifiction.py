import json
from datetime import datetime
from ultralytics import YOLO
import os

PT_MODEL    = "./runs/detect/train/weights/best.pt"
TRT_ENGINE  = "./runs/detect/train/weights/best.engine"
DATA_YAML   = "./dataset/detector_dataset/Weapons_Detection_New_v1_version_1_yolo26/data.yaml"
JSON_OUTPUT = "verification_reports/yolo_verification.json"


def save_json(data: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"💾 Metrics saved to {path}")


if __name__ == '__main__':

    COMMON_VAL = dict(
        data    = DATA_YAML,
        imgsz   = 640,
        split   = "val",
        batch   = 1,        # same batch for fair comparison
        rect    = False,    # no rect padding on either side
        device  = 0,        # pin to GPU 0 explicitly
        verbose = True,
    )

    # ── Baseline: original .pt (FP32)
    print("=" * 59)
    print("BASELINE — Original .pt model (FP32)")
    print("=" * 59)
    pt_model   = YOLO(PT_MODEL)
    pt_metrics = pt_model.val(**COMMON_VAL)
    pt_map50   = pt_metrics.box.map50
    pt_map5095 = pt_metrics.box.map

    pt_per_class = {}
    for i, name in enumerate(pt_metrics.names.values()):
        pt_per_class[name] = {
            "mAP50"   : round(float(pt_metrics.box.ap50[i]), 4),
            "mAP50-95": round(float(pt_metrics.box.ap[i]),   4),
        }

    # ── TRT FP16 Engine
    print("\n" + "=" * 59)
    print("TRT ENGINE — FP16 .engine model")
    print("=" * 59)
    trt_model   = YOLO(TRT_ENGINE, task="detect")
    trt_metrics = trt_model.val(**COMMON_VAL)   # no half=True needed
    trt_map50   = trt_metrics.box.map50
    trt_map5095 = trt_metrics.box.map

    trt_per_class = {}
    for i, name in enumerate(trt_metrics.names.values()):
        trt_per_class[name] = {
            "mAP50"   : round(float(trt_metrics.box.ap50[i]), 4),
            "mAP50-95": round(float(trt_metrics.box.ap[i]),   4),
        }

    # ── Drop calculation
    drop_map50   = round(float(pt_map50   - trt_map50),   4)
    drop_map5095 = round(float(pt_map5095 - trt_map5095), 4)

    per_class_drop = {
        name: {
            "mAP50_drop"   : round(pt_per_class[name]["mAP50"]    - trt_per_class[name]["mAP50"],    4),
            "mAP50-95_drop": round(pt_per_class[name]["mAP50-95"] - trt_per_class[name]["mAP50-95"], 4),
        }
        for name in pt_per_class
    }

    # ── Verdict
    if drop_map50 <= 0.01:
        verdict = "✅ Drop ≤ 1% — FP16 engine is safe for production"
    elif drop_map50 <= 0.02:
        verdict = "⚠️ Drop 1–2% — Acceptable but monitor on real footage"
    else:
        verdict = "❌ Drop > 2% — Consider FP32 engine or retrain with L2"

    # ── Print summary
    print("\n" + "=" * 59)
    print("COMPARISON SUMMARY")
    print("=" * 59)
    print(f"{'Metric':<20} {'PT (FP32)':>12} {'TRT FP16':>12} {'Drop':>10}")
    print("-" * 59)
    print(f"{'mAP@0.5':<20} {pt_map50:>12.4f} {trt_map50:>12.4f} {drop_map50:>+10.4f}")
    print(f"{'mAP@0.5:0.95':<20} {pt_map5095:>12.4f} {trt_map5095:>12.4f} {drop_map5095:>+10.4f}")
    print("-" * 59)
    print(verdict)

    report = {
        "timestamp" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": "YOLO Detection",
        "pt_model"  : PT_MODEL,
        "trt_engine": TRT_ENGINE,
        "test_split": "test",
        "val_config": {"imgsz": 640, "batch": 1, "rect": False, "device": 0},
        "verdict"   : verdict,
        "overall": {
            "pt_fp32" : {"mAP50": round(float(pt_map50),   4), "mAP50-95": round(float(pt_map5095), 4)},
            "trt_fp16": {"mAP50": round(float(trt_map50),  4), "mAP50-95": round(float(trt_map5095), 4)},
            "drop"    : {"mAP50": drop_map50, "mAP50-95": drop_map5095},
        },
        "per_class": {
            name: {
                "pt_fp32" : pt_per_class[name],
                "trt_fp16": trt_per_class[name],
                "drop"    : per_class_drop[name],
            }
            for name in pt_per_class
        }
    }

    save_json(report, JSON_OUTPUT)