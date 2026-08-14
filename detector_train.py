import os
import pandas as pd
import torch
import numpy as np
import random
import json
from ultralytics import YOLO
from ultralytics.data.augment import Albumentations
from config import *

# Set random seeds for reproducibility
def set_seeds(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==============================
# Augmentations (blur)
# ==============================
weapon_blur_augment = Albumentations(
    transforms=[
        {"type": "GaussBlur", "blur_limit": (3, 7), "p": 0.15},
        {"type": "MotionBlur", "blur_limit": (3, 7), "p": 0.10}
    ]
)

# ==============================
# Training function
# ==============================
def train_model():
    """Train YOLO model"""
    set_seeds()

    try:
        # Load YOLO model (use pretrained weights if available, otherwise default to YOLOv26)
        if os.path.exists(WEIGHTS):
            model = YOLO(WEIGHTS)
            print(f"✅ Loaded custom weights from {WEIGHTS}")
        else:
            model = YOLO("yolo26s.pt")  # Use YOLOv26 small as default
            print("✅ Loaded pretrained YOLOv26 small weights")

        # Training
        print("🚀 Starting training...")
        results = model.train(
            data=DATA_YAML,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            epochs=EPOCHS,
            lr0=LR_INITIAL,
            lrf=LR_FINAL,
            warmup_epochs=3.0,
            patience=PATIENCE,
            device=DEVICE,
            project=RESULTS_DIR,
            name="train",   
            cache="disk",
            optimizer="MuSGD",
            # freeze=10,
            val=True,
            save=True,
            # save_period=35,
            plots=True,  # YOLO will automatically save all plots
            verbose=True,
            exist_ok=True,
            cos_lr=True,
            cls=1.5,
            seed=RANDOM_SEED,
            workers=8,  
            amp=True,
            # pin_memory=True,
            # persistent_workers=True,
            # prefetch_factor=4,
            auto_augment=False,
            augment=True,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.5,
            degrees=15.0,
            translate=0.1,
            scale=0.5,
            shear=2.0,
            perspective=0.0008,
            copy_paste=0.1,
            #copy_paste_mode="mixup",
            mixup=0.0, 
            fliplr=0.4,
            mosaic=1.0,
            erasing=0.1,
            #albu=weapon_blur_augment,
            close_mosaic=20
        )

        print("✅ Training completed successfully.")

        # Save best epoch metrics
        results_dir = results.save_dir
        results_file = os.path.join(results_dir, "results.csv")

        if os.path.exists(results_file):
            df = pd.read_csv(results_file)
            df.columns = df.columns.str.strip()

            # Find the index (row) of the best epoch based on highest mAP50-95
            best_idx = df["metrics/mAP50-95(B)"].idxmax()
            best_row = df.loc[best_idx].to_dict()

            # Extract precision and recall
            precision = best_row.get("metrics/precision(B)", None)
            recall = best_row.get("metrics/recall(B)", None)

            # Compute F1 score
            if "metrics/f1(B)" in df.columns:
                f1_score = best_row.get("metrics/f1(B)", None)
            else:
                if precision and recall and (precision + recall) > 0:
                    f1_score = 2 * precision * recall / (precision + recall)
                else:
                    f1_score = None

            # Compile metrics dictionary
            metrics = {
                "epoch": int(best_row.get("epoch", 0)),
                "precision": float(precision) if precision is not None else None,
                "recall": float(recall) if recall is not None else None,
                "f1_score": float(f1_score) if f1_score is not None else None,
                "mAP50": float(best_row.get("metrics/mAP50(B)", 0)),
                "mAP50-95": float(best_row.get("metrics/mAP50-95(B)", 0)),
                "best_weights": os.path.join(results_dir, "weights", "best.pt"),
                "last_weights": os.path.join(results_dir, "weights", "last.pt"),
            }

            # Save metrics to TRAIN_METADATA_FILE
            os.makedirs(os.path.dirname(TRAIN_METADATA_FILE), exist_ok=True)
            with open(TRAIN_METADATA_FILE, "w") as f:
                json.dump(metrics, f, indent=4)

            print(f"✅ Best epoch metrics saved to {TRAIN_METADATA_FILE}")
            print(f"\n📊 Best Model Performance:")
            print(f"   Epoch: {metrics['epoch']}")
            print(f"   Precision: {metrics['precision']:.4f}" if metrics['precision'] else "   Precision: N/A")
            print(f"   Recall: {metrics['recall']:.4f}" if metrics['recall'] else "   Recall: N/A")
            print(f"   F1 Score: {metrics['f1_score']:.4f}" if metrics['f1_score'] else "   F1 Score: N/A")
            print(f"   mAP@0.5: {metrics['mAP50']:.4f}")
            print(f"   mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
            print(f"\n📁 Training results saved to: {results_dir}")
            print(f"   - All plots are in: {results_dir}")
            print(f"   - Best weights: {metrics['best_weights']}")
            print(f"   - Last weights: {metrics['last_weights']}")

        else:
            print("⚠️ No results.csv found. Could not save training metadata.")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


# ==============================
# MAIN EXECUTION
# ==============================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting YOLOv26 Training Pipeline")
    print("=" * 60)
    train_model()
    print("=" * 60)
    print("✅ Training pipeline completed successfully!")
    print("=" * 60)