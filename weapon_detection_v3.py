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
VIDEO_DIR        = "./test_videos"
YOLO_ENGINE_PATH = "./models/detector/best.engine"
CLF_ENGINE_PATH  = "./models/classifier/classifier_fp16.engine"
os.makedirs("output_videos_1", exist_ok=True)  
OUTPUT_DIR       = "./output_videos_1"  # set None to disable saving

IMG_SIZE         = 640
NUM_CLASSES      = 6

# ── Detection thresholds
CONF_THRES       = 0.40   # YOLO entry gate
CLF_CONF_THRES   = 0.55   # Min classifier confidence to accept a detection
DRAW_CONF_THRES  = 0.55   # Final draw gate (fused score)
NMS_IOU_THRES    = 0.40   # IoU threshold for post-classifier NMS

# ── Confidence fusion weight (YOLO share; classifier gets 1 - FUSION_ALPHA)
FUSION_ALPHA     = 0.45

# ── Crop quality filter
MIN_CROP_AREA    = 1500   # Reject tiny / edge crops (px²)

# ── Per-track classifier smoothing
#    Average the last N probability vectors per track before deciding label.
#    Higher = smoother labels, slightly more lag on fast class changes.
CLF_SMOOTH_LEN   = 5

# ── Temporal consistency via tracker
#    A detection is drawn only after the track has been confirmed
#    for at least TRACK_CONFIRM_FRAMES frames.
TRACK_CONFIRM_FRAMES = 3

# ── Stale track cleanup: purge track state if unseen for this many frames
STALE_TRACK_FRAMES   = 30

# ── Label agreement mode: "strict" | "soft" | "off"
LABEL_AGREEMENT  = "soft"
SOFT_AGREE_THRES = 0.70

# ── YOLO built-in tracker config ("bytetrack.yaml" or "botsort.yaml")
TRACKER_CONFIG   = "bytetrack.yaml"

# ── Frame skip for classifier (run classifier every N frames; use cached result otherwise)
#    Set to 1 to disable skipping. Increase for faster throughput on weak GPUs.
CLF_FRAME_SKIP   = 1


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
# PER-TRACK STATE
# ================================================================
class TrackState:
    """
    Holds per-track history for one tracked object.

    clf_probs_history : rolling deque of classifier probability vectors.
                        Averaged before each label decision — kills per-frame
                        flicker almost entirely.
    confirmed_frames  : how many consecutive frames this track has been seen
                        with a weapon label. Draw only once >= TRACK_CONFIRM_FRAMES.
    last_seen_frame   : absolute frame index of last update; used for stale cleanup.
    cached_clf_result : (label, conf) from the last classifier run, reused on
                        skipped frames.
    """
    __slots__ = (
        "clf_probs_history",
        "confirmed_frames",
        "last_seen_frame",
        "cached_clf_result",
    )

    def __init__(self):
        self.clf_probs_history: deque = deque(maxlen=CLF_SMOOTH_LEN)
        self.confirmed_frames: int    = 0
        self.last_seen_frame: int     = -1
        self.cached_clf_result        = None   # (label, conf) | None

    def update_probs(self, probs: np.ndarray):
        self.clf_probs_history.append(probs)

    def smoothed_probs(self) -> np.ndarray:
        """Return the element-wise mean of all stored probability vectors."""
        return np.mean(np.stack(self.clf_probs_history), axis=0)

    def increment_confirmed(self):
        self.confirmed_frames = min(self.confirmed_frames + 1, TRACK_CONFIRM_FRAMES + 10)

    def reset_confirmed(self):
        self.confirmed_frames = 0

    @property
    def is_confirmed(self) -> bool:
        return self.confirmed_frames >= TRACK_CONFIRM_FRAMES


class TrackStore:
    """
    Thread-safe(ish) registry of all active TrackState objects.
    Automatically purges tracks that haven't been seen in STALE_TRACK_FRAMES frames.
    """

    def __init__(self):
        self._store: dict[int, TrackState] = {}

    def get_or_create(self, track_id: int) -> TrackState:
        if track_id not in self._store:
            self._store[track_id] = TrackState()
        return self._store[track_id]

    def purge_stale(self, current_frame: int):
        """Remove tracks that have not been seen for STALE_TRACK_FRAMES frames."""
        stale = [
            tid for tid, ts in self._store.items()
            if (current_frame - ts.last_seen_frame) > STALE_TRACK_FRAMES
        ]
        for tid in stale:
            del self._store[tid]

    def active_count(self) -> int:
        return len(self._store)

    def reset(self):
        self._store.clear()


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

        self._warmup()
        print(f"✅ Classifier TRT engine loaded & warmed up: {engine_path}")

    # ── Internal helpers ──────────────────────────────────────────

    def _warmup(self, n: int = 3):
        """
        Run N dummy inferences to warm up the TRT execution context.
        Avoids the ~200 ms cold-start penalty on the first real frame.
        """
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        for _ in range(n):
            self.infer(dummy)

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        """BGR crop → ImageNet-normalized FP32 CHW tensor."""
        rgb  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        rgb  = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb  = (rgb - mean) / std
        return np.ascontiguousarray(rgb.transpose(2, 0, 1)[np.newaxis])

    # ── Public API ────────────────────────────────────────────────

    def infer(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        Run one crop through the classifier.
        Returns softmax probabilities of shape (NUM_CLASSES,).
        """
        self.input_tensor.copy_(torch.from_numpy(self._preprocess(crop_bgr)))
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
    Weighted fusion: FUSION_ALPHA × YOLO + (1 − FUSION_ALPHA) × classifier.
    Higher classifier weight since it acts as the false-positive rejection guard.
    """
    return FUSION_ALPHA * yolo_conf + (1 - FUSION_ALPHA) * clf_conf


# ================================================================
# POST-CLASSIFIER NMS
# ================================================================
def apply_nms(detections: list, iou_threshold: float = NMS_IOU_THRES) -> list:
    """
    Non-Maximum Suppression on classifier-confirmed detections.
    Signature: list of (x1, y1, x2, y2, label, conf, track_id)
    """
    if not detections:
        return []

    boxes  = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2, _, _, _ in detections]
    scores = [conf for *_, conf, _ in detections]

    keep = cv2.dnn.NMSBoxes(boxes, scores, DRAW_CONF_THRES, iou_threshold)

    if len(keep) == 0:
        return []

    return [detections[i] for i in keep.flatten()]


# ================================================================
# LABEL AGREEMENT CHECK
# ================================================================
def passes_label_agreement(yolo_label: str, clf_label: str, clf_conf: float) -> bool:
    """
    Return True if YOLO and classifier agree enough to accept this detection.
    Controlled by the LABEL_AGREEMENT config constant.
    """
    if LABEL_AGREEMENT == "off":
        return True
    if LABEL_AGREEMENT == "strict":
        return clf_label == yolo_label
    if LABEL_AGREEMENT == "soft":
        # Reject only when the classifier is CONFIDENTLY pointing at a different class.
        if clf_label != yolo_label and clf_conf > SOFT_AGREE_THRES:
            return False
        return True
    return True


# ================================================================
# OVERLAY HELPERS
# ================================================================
def draw_hud(frame: np.ndarray, source_name: str, weapon_counts: dict, fps: float):
    """
    Draw source name, per-class weapon counts, and live FPS in a corner HUD.
    """
    h, w = frame.shape[:2]

    # Source label (top-left)
    cv2.putText(frame, source_name, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # FPS (top-right)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 140, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Per-class counts (bottom-left) — only show classes with at least 1 detection
    y_offset = h - 20
    for cls_name, count in weapon_counts.items():
        if count > 0:
            color = CLASS_COLORS.get(cls_name, (255, 255, 255))
            cv2.putText(frame, f"{cls_name}: {count}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset -= 28


def draw_detection(frame, x1, y1, x2, y2, label, conf, track_id):
    """Draw a single bounding box with label, confidence, and track ID."""
    color = CLASS_COLORS.get(label, (255, 255, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cvzone.putTextRect(
        frame,
        f"[{track_id}] {label}  {conf:.2f}",
        (x1, max(y1 - 10, 20)),
        scale=1.1,
        thickness=2,
        colorB=color,
        colorT=(255, 255, 255),
        offset=3,
    )


# ================================================================
# VIDEO WRITER FACTORY
# ================================================================
def make_writer(output_path: str, fps: float, frame_size: tuple) -> cv2.VideoWriter | None:
    if output_path is None:
        return None
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    if not writer.isOpened():
        print(f"⚠️  Could not open VideoWriter for: {output_path}")
        return None
    return writer


# ================================================================
# MAIN INFERENCE LOOP
# ================================================================
def run_inference(video_sources=None):

    print("📦 Loading YOLO TRT engine...")
    yolo_model = YOLO(YOLO_ENGINE_PATH, task="detect")
    print("✅ YOLO TRT engine loaded\n")

    clf_engine  = ClassifierEngine(CLF_ENGINE_PATH)
    track_store = TrackStore()

    if video_sources is None:
        print("🎥 No videos found — switching to webcam")
        sources      = [cv2.VideoCapture(0)]
        source_names = ["Webcam"]
    else:
        sources      = [cv2.VideoCapture(v) for v in video_sources]
        source_names = [os.path.basename(v) for v in video_sources]

    WINDOW = "Weapon Detection — YOLO TRT + Classifier TRT [v3]"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.moveWindow(WINDOW, 200, 80)

    exit_all = False

    for cap, name in zip(sources, source_names):

        if not cap.isOpened():
            print(f"⚠️  Could not open: {name}")
            continue

        src_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        delay      = max(int(1000 / src_fps), 1)
        print(f"\n▶  Playing: {name}  ({frame_w}×{frame_h} @ {src_fps:.1f} fps)")

        # Reset per-video state
        track_store.reset()

        # Optional video writer
        writer = None
        if OUTPUT_DIR is not None:
            out_path = os.path.join(OUTPUT_DIR, f"out_{name}")
            writer   = make_writer(out_path, src_fps, (frame_w, frame_h))

        frame_index = 0        # absolute frame counter
        seek_target = None     # set by 'a'/'d' keypresses
        prev_tick   = cv2.getTickCount()

        while True:

            # ── Seek when user pressed a/d ──────────────────────
            if seek_target is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(seek_target, 0))
                frame_index  = max(seek_target, 0)
                seek_target  = None
                # Reset tracker state after a seek — stale IDs are meaningless
                yolo_model.predictor = None   # forces tracker re-init on next call
                track_store.reset()

            success, frame = cap.read()
            if not success:
                break

            # ── Live FPS measurement ─────────────────────────────
            cur_tick  = cv2.getTickCount()
            live_fps  = cv2.getTickFrequency() / (cur_tick - prev_tick + 1e-9)
            prev_tick = cur_tick

            # ── YOLO tracking (replaces predict + manual TemporalTracker) ──
            #
            #    persist=True  → tracker keeps its internal state between calls,
            #                    so the same real-world object keeps the same ID.
            #    Both YOLO and the tracker are running from the same .engine;
            #    no extra overhead vs the old .predict() call.
            # ────────────────────────────────────────────────────────────────
            results = yolo_model.track(
                frame,
                imgsz=IMG_SIZE,
                conf=CONF_THRES,
                persist=True,
                tracker=TRACKER_CONFIG,
                verbose=False,
            )

            raw_detections = []

            for r in results:
                for box in r.boxes:

                    # ── 1. Parse YOLO box ────────────────────────
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id          = int(box.cls[0])
                    yolo_conf       = float(box.conf[0])
                    yolo_label      = WEAPON_CLASSES[cls_id]

                    # Track ID from ByteTrack/BoTSORT.
                    # box.id can be None if the tracker hasn't assigned an ID yet
                    # (typically on the very first frame or after an occlusion gap).
                    # In that case we fall back to a negative pseudo-ID so the
                    # detection still passes through without corrupting real track state.
                    track_id = int(box.id[0]) if box.id is not None else -(frame_index)

                    # ── 2. Crop extraction ───────────────────────
                    crop       = frame[y1:y2, x1:x2]
                    crop_area  = crop.shape[0] * crop.shape[1] if crop.size > 0 else 0

                    if crop.size == 0 or crop_area < MIN_CROP_AREA:
                        continue

                    # ── 3. Per-track state ───────────────────────
                    ts = track_store.get_or_create(track_id)
                    ts.last_seen_frame = frame_index

                    # ── 4. Classifier inference (with frame-skip cache) ──
                    run_clf = (CLF_FRAME_SKIP <= 1) or (frame_index % CLF_FRAME_SKIP == 0)

                    if run_clf:
                        probs = clf_engine.infer(crop)
                        ts.update_probs(probs)
                        # Cache result for skipped frames
                        smoothed   = ts.smoothed_probs()
                        pred_idx   = int(np.argmax(smoothed))
                        clf_conf   = float(smoothed[pred_idx])
                        clf_label  = WEAPON_CLASSES[pred_idx]
                        ts.cached_clf_result = (clf_label, clf_conf)
                    else:
                        # Reuse last known result
                        if ts.cached_clf_result is None:
                            continue   # no classifier result yet — skip this frame
                        clf_label, clf_conf = ts.cached_clf_result

                    # ── 5. Suppress non_weapon predictions ───────
                    if clf_label == "non_weapon":
                        ts.reset_confirmed()
                        continue

                    # ── 6. Minimum classifier confidence gate ─────
                    if clf_conf < CLF_CONF_THRES:
                        ts.reset_confirmed()
                        continue

                    # ── 7. Label agreement check ──────────────────
                    if not passes_label_agreement(yolo_label, clf_label, clf_conf):
                        ts.reset_confirmed()
                        continue

                    # ── 8. Determine final label ──────────────────
                    final_label = clf_label if clf_conf >= SOFT_AGREE_THRES else yolo_label

                    # ── 9. Confidence fusion ──────────────────────
                    final_conf = fuse_confidence(yolo_conf, clf_conf)

                    # ── 10. Per-track confirmation counter ────────
                    #
                    #    Old approach: TemporalTracker checked whether a *class label*
                    #    appeared in ≥N of the last 5 frames — susceptible to label
                    #    flicker inflating the count.
                    #
                    #    New approach: we increment a per-track counter only when this
                    #    specific track_id passes all gates. Because ByteTrack assigns a
                    #    stable ID to the same physical object, this is a true
                    #    "has this specific weapon been consistently detected" check.
                    # ─────────────────────────────────────────────
                    ts.increment_confirmed()

                    raw_detections.append(
                        (x1, y1, x2, y2, final_label, final_conf, track_id)
                    )

            # ── 11. Post-classifier NMS ──────────────────────────
            detections = apply_nms(raw_detections)

            # ── 12. Stale track cleanup (every 30 frames) ────────
            if frame_index % 30 == 0:
                track_store.purge_stale(frame_index)

            # ── Draw confirmed detections ─────────────────────────
            weapon_counts = defaultdict(int)

            for x1, y1, x2, y2, label, conf, track_id in detections:

                if conf < DRAW_CONF_THRES:
                    continue

                ts = track_store.get_or_create(track_id)
                if not ts.is_confirmed:
                    continue

                weapon_counts[label] += 1
                draw_detection(frame, x1, y1, x2, y2, label, conf, track_id)

            # ── HUD overlay ──────────────────────────────────────
            draw_hud(frame, name, weapon_counts, live_fps)

            # ── Write to file if enabled ─────────────────────────
            if writer is not None:
                writer.write(frame)

            cv2.imshow(WINDOW, frame)
            frame_index += 1

            # ── Keyboard controls ────────────────────────────────
            #   q  → next video
            #   e  → exit all
            #   d  → skip forward  30 frames
            #   a  → skip backward 30 frames
            #   s  → screenshot
            key = cv2.waitKey(delay) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('e'):
                exit_all = True
                break
            elif key == ord('d'):
                seek_target = frame_index + 30
            elif key == ord('a'):
                seek_target = frame_index - 30
            elif key == ord('s'):
                ss_path = os.path.join(
                    OUTPUT_DIR or ".", f"screenshot_{name}_{frame_index:06d}.jpg"
                )
                cv2.imwrite(ss_path, frame)
                print(f"📸 Screenshot saved: {ss_path}")

        # ── Cleanup for this video ───────────────────────────────
        cap.release()
        if writer is not None:
            writer.release()
        track_store.reset()

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
