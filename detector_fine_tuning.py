import os
import pandas as pd
import torch
import numpy as np
import random
import json
from ultralytics import YOLO
from config_fine_tune   import *

# ==============================
# Seed
# ==============================
def set_seeds(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==============================
# Fine-tuning function
# ==============================
def finetune_model():
    """Fine-tune YOLO model from best.pt weights""" 
    set_seeds()

    # ── Resolve weights path ──────────────────────────────────────
    if os.path.exists(BEST_WEIGHTS):
        weights_path = BEST_WEIGHTS
        print(f"✅ Fine-tuning from best weights: {weights_path}")
    elif os.path.exists(LAST_WEIGHTS):
        weights_path = LAST_WEIGHTS
        print(f"⚠️ best.pt not found, falling back to last.pt: {weights_path}")
    else:
        raise FileNotFoundError(
            f"❌ No weights found at:\n  {BEST_WEIGHTS}\n  {LAST_WEIGHTS}"
        )

    try:
        model = YOLO(weights_path)

        print("🔧 Starting fine-tuning...")
        results = model.train(
            data=DATA_YAML,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            epochs=EPOCHS,                  # Shorter than full training
            lr0=LR_INITIAL,                  # Much lower LR — critical for fine-tuning
            lrf=LR_FINAL,                    # Gentle final LR decay
            warmup_epochs=1.0,          # Less warmup needed
            patience=PATIENCE,                # More patience for fine-tuning
            device=DEVICE,
            project=RESULTS_DIR,
            name="finetune",            # Separate from original "train" run
            cache="disk",
            optimizer="MuSGD",
            freeze=10,                  # ← Freeze first 10 layers; preserves
                                        #   learned low-level features, only
                                        #   fine-tunes deeper layers
            val=True,
            save=True,
            plots=True,
            verbose=True,
            exist_ok=True,
            cos_lr=True,
            cls=1.5,
            seed=RANDOM_SEED,
            workers=8,
            amp=True,
            auto_augment=False,
            augment=True,               # Keep augmentation but lighter
            hsv_h=0.01,                 # Slightly reduced from training
            hsv_s=0.5,                  # Reduced
            hsv_v=0.4,                  # Reduced
            degrees=10.0,               # Less rotation for fine-tuning
            translate=0.05,             # Less translate
            scale=0.3,                  # Less scale
            shear=1.0,                  # Less shear
            perspective=0.0005,
            copy_paste=0.05,            # Reduced
            mixup=0.0,
            fliplr=0.4,
            mosaic=0.5,                 # Reduced mosaic for fine-tuning
            erasing=0.1,
            close_mosaic=10,            # Earlier mosaic close
        )

        print("✅ Fine-tuning completed successfully.")

        # ── Save best epoch metrics ───────────────────────────────
        results_dir = results.save_dir
        results_file = os.path.join(results_dir, "results.csv")

        if os.path.exists(results_file):
            df = pd.read_csv(results_file)
            df.columns = df.columns.str.strip()

            best_idx = df["metrics/mAP50-95(B)"].idxmax()
            best_row = df.loc[best_idx].to_dict()

            precision = best_row.get("metrics/precision(B)", None)
            recall = best_row.get("metrics/recall(B)", None)

            if "metrics/f1(B)" in df.columns:
                f1_score = best_row.get("metrics/f1(B)", None)
            else:
                if precision and recall and (precision + recall) > 0:
                    f1_score = 2 * precision * recall / (precision + recall)
                else:
                    f1_score = None

            metrics = {
                "epoch": int(best_row.get("epoch", 0)),
                "precision": float(precision) if precision is not None else None,
                "recall": float(recall) if recall is not None else None,
                "f1_score": float(f1_score) if f1_score is not None else None,
                "mAP50": float(best_row.get("metrics/mAP50(B)", 0)),
                "mAP50-95": float(best_row.get("metrics/mAP50-95(B)", 0)),
                "fine_tuned_from": weights_path,
                "best_weights": os.path.join(results_dir, "weights", "best.pt"),
                "last_weights": os.path.join(results_dir, "weights", "last.pt"),
            }

            # Save to a separate finetune metadata file
            finetune_metadata_file = os.path.join(
                RESULTS_DIR, "./runs/detect/finetuned", "finetune_metadata.json"
            )
            os.makedirs(os.path.dirname(finetune_metadata_file), exist_ok=True)
            with open(finetune_metadata_file, "w") as f:
                json.dump(metrics, f, indent=4)

            print(f"✅ Fine-tune metrics saved to {finetune_metadata_file}")
            print(f"\n📊 Best Fine-Tuned Model Performance:")
            print(f"   Epoch:     {metrics['epoch']}")
            print(f"   Precision: {metrics['precision']:.4f}" if metrics['precision'] else "   Precision: N/A")
            print(f"   Recall:    {metrics['recall']:.4f}" if metrics['recall'] else "   Recall: N/A")
            print(f"   F1 Score:  {metrics['f1_score']:.4f}" if metrics['f1_score'] else "   F1 Score: N/A")
            print(f"   mAP@0.5:      {metrics['mAP50']:.4f}")
            print(f"   mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
            print(f"\n📁 Fine-tune results saved to: {results_dir}")
            print(f"   - Best weights: {metrics['best_weights']}")
            print(f"   - Last weights: {metrics['last_weights']}")

        else:
            print("⚠️ No results.csv found. Could not save fine-tune metadata.")

    except Exception as e:
        print(f"❌ Fine-tuning failed: {e}")
        raise


# ==============================
# MAIN EXECUTION
# ==============================
if __name__ == "__main__":
    print("=" * 60)
    print("🔧 Starting YOLO26 Fine-Tuning Pipeline")
    print("=" * 60)
    finetune_model()
    print("=" * 60)
    print("✅ Fine-tuning pipeline completed successfully!")
    print("=" * 60)