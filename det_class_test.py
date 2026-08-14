import os
import glob
import cv2
import cvzone
import numpy as np
import torch
import tensorrt as trt

from collections import defaultdict, deque
from ultralytics import YOLO

# ================================================================
# CONFIG
# ================================================================
VIDEO_DIR        = r"C:\Users\Shakiru Sikiru\ML\Weapon_Detection\v26\test_videos"
YOLO_ENGINE_PATH = r"C:\Users\Shakiru Sikiru\ML\Weapon_Detection\v26\runs\detect\train\weights\best.engine"
CLF_ENGINE_PATH  = r"C:\Users\Shakiru Sikiru\ML\Weapon_Detection\v26\classifier_reports\classifier_fp16.engine"

IMG_SIZE         = 640
NUM_CLASSES      = 6

# ── Thresholds (raised from v1 to reduce false positives)
CONF_THRES       = 0.40   # ↑ was 0.25 — YOLO entry gate
CLF_CONF_THRES   = 0.55   # NEW — minimum classifier confidence to accept
DRAW_CONF_THRES  = 0.55   # ↑ was 0.40 — final draw gate
NMS_IOU_THRES    = 0.40   # NEW — IoU threshold for post-classifier NMS

# ── Fusion: give MORE weight to classifier since it's the FP guard
FUSION_ALPHA     = 0.45   # ↓ was 0.60 (0.45 YOLO + 0.55 classifier)

# ── Crop quality filter
MIN_CROP_AREA    = 1500   # NEW — reject tiny / edge crops (px²)

# ── Temporal consistency (video)
TEMPORAL_WINDOW  = 3      # NEW — detections must appear in N of last 5 frames
TEMPORAL_MAXLEN  = 5

# ── Label agreement mode: "strict" | "soft" | "off"
#    strict → YOLO and classifier must agree on exact class
#    soft   → suppress only when classifier is confident about a DIFFERENT class
#    off    → classifier label always wins (original behaviour)
LABEL_AGREEMENT  = "soft"
SOFT_AGREE_THRES = 0.70   # classifier confidence required to override YOLO label


# ================================================================
# CLASSES
# ================================================================
WEAPON_CLASSES = [
    "Handgun",
    "Heavy Weapon",
    "Knife",
    "Rifle",
    "Rocket_Grenade Launcher",
    "non_weapon",
]

CLASS_COLORS = {
    "Handgun"                : (128, 0,   128),
    "Heavy Weapon"           : (255, 0,   0  ),
    "Knife"                  : (0,   255, 0  ),
    "Rifle"                  : (0,   0,   255),
    "Rocket_Grenade Launcher": (0,   165, 255),
}


# ================================================================
# CLASSIFIER TRT ENGINE
# ================================================================
class ClassifierEngine:
    """
    TRT FP16 classifier using PyTorch CUDA tensors for bindings.
    Shares the same CUDA context as Ultralytics YOLO — no context conflict.
    All weapon classes pass through this classifier.
    non_weapon output → detection suppressed as false positive.
    """

    def __init__(self, engine_path: str):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine  = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_tensor  = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device="cuda")
        self.output_tensor = torch.zeros((1, NUM_CLASSES),  dtype=torch.float32, device="cuda")

        print(f"✅ Classifier TRT engine loaded: {engine_path}")

    def preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        """BGR crop → normalized FP32 CHW tensor (ImageNet stats)."""
        rgb  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        rgb  = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb  = (rgb - mean) / std
        return np.ascontiguousarray(rgb.transpose(2, 0, 1)[np.newaxis])

    def infer(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        Run one crop through classifier.
        Returns softmax probabilities, shape (NUM_CLASSES,).
        """
        self.input_tensor.copy_(torch.from_numpy(self.preprocess(crop_bgr)))
        self.context.execute_v2(bindings=[
            self.input_tensor.data_ptr(),
            self.output_tensor.data_ptr(),
        ])
        torch.cuda.synchronize()

        logits = self.output_tensor[0].cpu().numpy()
        exp    = np.exp(logits - logits.max())
        return exp / exp.sum()


# ================================================================
# CONFIDENCE FUSION
# ================================================================
def fuse_confidence(yolo_conf: float, clf_conf: float) -> float:
    """
    Weighted fusion: FUSION_ALPHA * YOLO + (1 - FUSION_ALPHA) * classifier.
    Higher classifier weight since it acts as the FP rejection guard.
    """
    return FUSION_ALPHA * yolo_conf + (1 - FUSION_ALPHA) * clf_conf


# ================================================================
# POST-CLASSIFIER NMS
# ================================================================
def apply_nms(detections: list, iou_threshold: float = NMS_IOU_THRES) -> list:
    """
    Apply Non-Maximum Suppression to classifier-confirmed detections.
    Removes redundant overlapping boxes before drawing.
    detections: list of (x1, y1, x2, y2, label, conf)
    """
    if not detections:
        return []

    boxes  = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2, _, _ in detections]
    scores = [conf for *_, conf in detections]

    keep = cv2.dnn.NMSBoxes(boxes, scores, DRAW_CONF_THRES, iou_threshold)

    if len(keep) == 0:
        return []

    return [detections[i] for i in keep.flatten()]


# ================================================================
# TEMPORAL CONSISTENCY TRACKER
# ================================================================
class TemporalTracker:
    """
    Tracks per-class detection history over a sliding window.
    A detection is only confirmed if it appeared in at least
    TEMPORAL_WINDOW of the last TEMPORAL_MAXLEN frames.
    """

    def __init__(self, window: int = TEMPORAL_WINDOW, maxlen: int = TEMPORAL_MAXLEN):
        self.window  = window
        self.history = defaultdict(lambda: deque(maxlen=maxlen))

    def update(self, active_labels: set):
        """Call once per frame with the set of labels detected this frame."""
        # Record presence/absence for every known label seen so far
        all_labels = set(self.history.keys()) | active_labels
        for label in all_labels:
            self.history[label].append(1 if label in active_labels else 0)

    def is_confirmed(self, label: str) -> bool:
        """Return True if the label has been consistently detected recently."""
        return sum(self.history[label]) >= self.window


# ================================================================
# LABEL AGREEMENT CHECK
# ================================================================
def passes_label_agreement(yolo_label: str, clf_label: str, clf_conf: float) -> bool:
    """
    Check whether YOLO and classifier agree enough to accept this detection.
    Behaviour is controlled by LABEL_AGREEMENT config.
    """
    if LABEL_AGREEMENT == "off":
        return True

    if LABEL_AGREEMENT == "strict":
        return clf_label == yolo_label

    if LABEL_AGREEMENT == "soft":
        # Only reject when the classifier is CONFIDENTLY pointing at a different class
        if clf_label != yolo_label and clf_conf > SOFT_AGREE_THRES:
            return False
        return True

    return True


# ================================================================
# MAIN INFERENCE LOOP
# ================================================================
def run_inference(video_sources=None):

    print("📦 Loading YOLO TRT engine...")
    yolo_model = YOLO(YOLO_ENGINE_PATH, task="detect")
    print("✅ YOLO TRT engine loaded\n")

    clf_engine      = ClassifierEngine(CLF_ENGINE_PATH)
    temporal_tracker = TemporalTracker()

    if video_sources is None:
        print("🎥 No videos found — switching to webcam")
        sources      = [cv2.VideoCapture(0)]
        source_names = ["Webcam"]
    else:
        sources      = [cv2.VideoCapture(v) for v in video_sources]
        source_names = [os.path.basename(v) for v in video_sources]

    WINDOW = "Weapon Detection — YOLO TRT + Classifier TRT [v2]"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.moveWindow(WINDOW, 200, 80)

    exit_all    = False
    frame_index = 0

    for cap, name in zip(sources, source_names):

        if not cap.isOpened():
            print(f"⚠️  Could not open: {name}")
            continue

        fps   = cap.get(cv2.CAP_PROP_FPS)
        delay = max(int(1000 / fps) if fps > 0 else 1, 1)
        print(f"\n▶  Playing: {name}")

        # Reset temporal tracker per video
        temporal_tracker = TemporalTracker()

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = cap.read()
            if not success:
                break

            # ── YOLO detection ─────────────────────────────────────────
            results = yolo_model.predict(
                frame,
                imgsz=IMG_SIZE,
                conf=CONF_THRES,   # ↑ raised entry threshold
                verbose=False
            )

            raw_detections = []

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id          = int(box.cls[0])
                    yolo_conf       = float(box.conf[0])
                    yolo_label      = WEAPON_CLASSES[cls_id]

                    # ── 1. Crop extraction ──────────────────────────────
                    crop    = frame[y1:y2, x1:x2]
                    crop_h, crop_w = crop.shape[:2] if crop.size > 0 else (0, 0)

                    # ── 2. Skip empty / tiny crops ──────────────────────
                    if crop.size == 0 or (crop_h * crop_w) < MIN_CROP_AREA:
                        continue

                    # ── 3. Classifier inference ─────────────────────────
                    probs     = clf_engine.infer(crop)
                    pred_idx  = int(np.argmax(probs))
                    clf_conf  = float(probs[pred_idx])
                    clf_label = WEAPON_CLASSES[pred_idx]

                    # ── 4. Suppress non_weapon predictions ──────────────
                    if clf_label == "non_weapon":
                        continue

                    # ── 5. Minimum classifier confidence gate ────────────
                    if clf_conf < CLF_CONF_THRES:
                        continue

                    # ── 6. Label agreement check ─────────────────────────
                    if not passes_label_agreement(yolo_label, clf_label, clf_conf):
                        continue

                    # ── 7. Determine final label ─────────────────────────
                    #    In soft/off mode: prefer classifier label when confident,
                    #    fall back to YOLO label when classifier is uncertain
                    if clf_conf >= SOFT_AGREE_THRES:
                        final_label = clf_label
                    else:
                        final_label = yolo_label

                    # ── 8. Confidence fusion ─────────────────────────────
                    final_conf = fuse_confidence(yolo_conf, clf_conf)

                    raw_detections.append((x1, y1, x2, y2, final_label, final_conf))

            # ── 9. Post-classifier NMS ──────────────────────────────────
            detections = apply_nms(raw_detections)

            # ── 10. Temporal consistency update ─────────────────────────
            active_labels = {label for *_, label, _ in detections}
            temporal_tracker.update(active_labels)

            # ── Draw confirmed detections ───────────────────────────────
            for x1, y1, x2, y2, label, conf in detections:

                # Draw threshold gate
                if conf < DRAW_CONF_THRES:
                    continue

                # Temporal consistency gate
                if not temporal_tracker.is_confirmed(label):
                    continue

                color = CLASS_COLORS.get(label, (255, 255, 255))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cvzone.putTextRect(
                    frame,
                    f"{label}  {conf:.2f}",
                    (x1, max(y1 - 10, 20)),
                    scale=1.2,
                    thickness=2,
                    colorB=color,
                    colorT=(255, 255, 255),
                    offset=3
                )

            # ── Source label overlay
            cv2.putText(
                frame, name, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2
            )

            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('e'):
                exit_all = True
                break
            elif key == ord('d'):
                frame_index += 30
                continue
            elif key == ord('a'):
                frame_index = max(frame_index - 30, 0)
                continue
            else:
                frame_index += 1

        cap.release()
        frame_index = 0

        if exit_all:
            break

    cv2.destroyAllWindows()


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":

    extensions  = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    video_paths = []

    for ext in extensions:
        video_paths.extend(glob.glob(os.path.join(VIDEO_DIR, ext)))

    if not video_paths:
        run_inference(None)
    else:
        print(f"📁 Found {len(video_paths)} video(s)")
        run_inference(video_paths)