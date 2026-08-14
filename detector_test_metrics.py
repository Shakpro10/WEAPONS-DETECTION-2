import json
import os
from ultralytics import YOLO
from config import RESULTS_DIR, BEST_WEIGHTS, TEST_METADATA_FILE, DATA_YAML, BATCH_SIZE, IMG_SIZE, DEVICE

# Default to the weights from the training script's output location
best_weight = BEST_WEIGHTS
def validate_model(best_weights=best_weight):
    """Run validation and save metrics"""
    if not os.path.exists(best_weights):
        print(f"⚠️ {best_weights} not found, skipping validation.")
        return None
    
    try:
        # Load YOLOv26 model
        model = YOLO(best_weights)
        print(f"✅ Loaded YOLOv26 model from {best_weights}")

        # Run validation
        print("🚀 Starting validation on test set...")
        results = model.val(
            data=DATA_YAML,
            batch=BATCH_SIZE,
            imgsz=IMG_SIZE,
            device=DEVICE,
            split="test",
            name="test",
            save_json=True,
            exist_ok=True,
            verbose=True,
            plots=True  # YOLO will automatically save confusion matrix and other plots
        )

        # Extract metrics
        metrics = results.results_dict  # Dict with keys: precision, recall, map50, map
        precision = float(metrics.get("metrics/precision(B)", 0))
        recall = float(metrics.get("metrics/recall(B)", 0))
        mAP50 = float(metrics.get("metrics/mAP50(B)", 0))
        mAP50_95 = float(metrics.get("metrics/mAP50-95(B)", 0))
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-16)

        summary = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
            "mAP50": round(mAP50, 4),
            "mAP50_95": round(mAP50_95, 4),
            "weights_path": best_weights
        }

        # Save metrics to test_metadata.json
        os.makedirs(os.path.dirname(TEST_METADATA_FILE), exist_ok=True)
        with open(TEST_METADATA_FILE, "w") as f:
            json.dump(summary, f, indent=4)

        print(f"\n✅ Validation metrics saved to {TEST_METADATA_FILE}")
        print(f"\n📊 Test Set Performance:")
        print(f"   Precision: {summary['precision']:.4f}")
        print(f"   Recall: {summary['recall']:.4f}")
        print(f"   F1 Score: {summary['f1_score']:.4f}")
        print(f"   mAP@0.5: {summary['mAP50']:.4f}")
        print(f"   mAP@0.5:0.95: {summary['mAP50_95']:.4f}")
        
        return summary

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        raise


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting YOLOv26 Validation Pipeline")
    print("=" * 60)
    validate_model(best_weights=best_weight)
    print("=" * 60)
    print("✅ Validation complete!")
    print("=" * 60)