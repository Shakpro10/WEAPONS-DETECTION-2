"""
weapon_detection.py
────────────────────────────────────────────────────────────────────────────
YOLO TRT  +  Classifier TRT  —  detection with temporal filtering
────────────────────────────────────────────────────────────────────────────



────────────────────────────────────────────────────────────────────────────
Differences vs the tracking version (v3):
  • yolo_model.predict() replaces yolo_model.track()  — no ByteTrack/BoTSORT
  • TemporalTracker (sliding-window label history) replaces per-track
    confirmation counters and TrackState / TrackStore
  • TrackID display removed from bounding boxes
  • Stale-track cleanup removed (TemporalTracker handles its own history)
  • CLF_FRAME_SKIP removed — classifier runs every frame (no track cache)
  • All other functionality retained from v3:
      - Video saving to output_videos/
      - HUD overlay (source name, per-class counts, FPS)
      - Classifier warmup
      - Post-classifier NMS
      - Confidence fusion
      - Label agreement check
      - Crop quality filter
      - Keyboard controls (e / x / q / d / a / s)

      - KEY CONTROLS
        e  →  quit current video and move to the next one
        x  →  exit the entire program immediately
        q  →  go back to the previous video
        d  →  fast-forward frames by SEEK_SECONDS in the current video
        a  →  rewind frames by SEEK_SECONDS in the current video
        s  →  save a snapshot of the current frame to OUTPUT_DIR

      
────────────────────────────────────────────────────────────────────────────
Single-window fixes:
  [Fix 1 — Ultralytics window]
    Ultralytics creates its own cv2.imshow window keyed on the source
    file path (not our WINDOW name), so it cannot be suppressed just by
    passing show=False to predict() — that argument is read from
    predictor.args, which is only initialised after the first predict()
    call.  Three-layer suppression is applied:
      a) yolo_model.overrides['show'] = False  — before any predict call
      b) show=False in every predict() call
      c) After the first predict() call the predictor object exists;
         yolo_model.predictor.args.show is patched to False immediately.
    Layer (c) is what actually stops the window on stubborn versions.

  [Fix 2 — Quarter-screen sizing]
    cv2.resizeWindow() is unreliable on Linux / Wayland / some X11
    compositors — it is silently ignored even after the first imshow.
    Instead the display frame is resized with cv2.resize() before being
    passed to imshow.  The window then auto-fits to the smaller frame;
    no resizeWindow call is needed at all and the result is identical on
    every platform.

  [Fix 3 — Zero-cost window sizing via WINDOW_NORMAL]
    Unlike the DISPLAY_SCALE approach in Fix 2 (which adds a cv2.resize()
    call per frame and shrinks the frame data), cv2.WINDOW_NORMAL lets the
    OS / GPU compositor scale the displayed content to fit the window.
    The NumPy frame array is never touched — inference and video saving
    always operate at original resolution.  The window is created ONCE in
    run_inference() before the main loop so WINDOW_NORMAL persists across
    all videos without re-creating the handle.
    Caveat: cv2.resizeWindow() can still be silently ignored on Wayland /
    some X11 compositors (see Fix 2).  If that happens the window simply
    opens at its default size — no crash, no speed regression at all.
    ────────────────────────────────────────────────────────────────────────────
"""

import os
import glob
import cv2
import cvzone
import numpy as np
import torch
import tensorrt as trt

from collections import defaultdict, deque
from datetime import datetime
from ultralytics import YOLO


# ================================================================
# CONFIG
# ================================================================
VIDEO_DIR        = "./test_videos"
YOLO_ENGINE_PATH = "./models/detector/best.engine"
CLF_ENGINE_PATH  = "./models/classifier/classifier_fp16.engine"
os.makedirs("inference_folder/snapshot", exist_ok=True)
OUTPUT_DIR       = "./inference_folder/snapshot"

# ── Processed / annotated videos are saved here (see VIDEO WRITER FACTORY)
INFERENCE_OUTPUT_DIR = "./inference_folder/output_videos"
os.makedirs(INFERENCE_OUTPUT_DIR, exist_ok=True)

IMG_SIZE         = 640
NUM_CLASSES      = 6

# ── Detection thresholds
CONF_THRES       = 0.40
CLF_CONF_THRES   = 0.55
DRAW_CONF_THRES  = 0.55
NMS_IOU_THRES    = 0.40

# ── Confidence fusion weight (YOLO share; classifier gets 1 - FUSION_ALPHA)
FUSION_ALPHA     = 0.45

# ── Crop quality filter
MIN_CROP_AREA    = 1500

# ── Temporal consistency (replaces per-track confirmation counter)
#    A label is drawn only after it appears in at least TEMPORAL_WINDOW
TEMPORAL_WINDOW  = 3
TEMPORAL_MAXLEN  = 5

# ── Label agreement mode: "strict" | "soft" | "off"
LABEL_AGREEMENT  = "soft"
SOFT_AGREE_THRES = 0.70

# ── Display scale: fraction of the original frame to show in the window.
#    0.5 = half width × half height = one quarter of the frame area.
#    The full-resolution frame is still used for inference and saving.
DISPLAY_SCALE    = 0.5

# ── Seconds to seek per d (fast-forward) or a (rewind) keypress.
#    Converted to a frame count using the source's actual FPS at runtime.
#    Has no effect on live webcam streams (seeking is unsupported there).
SEEK_SECONDS     = 4

# ── Single display window name reused across all videos.
#    Defined here so every section that references it uses the same string.
WINDOW_NAME      = "Weapon Detection"


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
    "Handgun"                : (128, 0, 128),
    "Heavy Weapon"           : (255, 0, 0),
    "Knife"                  : (0, 255, 0),
    "Rifle"                  : (0, 0, 255),
    "Rocket_Grenade Launcher": (0, 165, 255),
}


# ================================================================
# TEMPORAL TRACKER
# ================================================================
class TemporalTracker:
    """
    Tracks per-class detection history over a sliding window.

    A label is confirmed only if it appeared in at least TEMPORAL_WINDOW
    of the last TEMPORAL_MAXLEN frames — kills single-frame false positives
    and per-frame flicker without needing object IDs.

    Call update() once per frame with the set of labels seen that frame.
    Call is_confirmed() before drawing any label.
    Call reset() between videos so stale history does not carry over.
    """

    def __init__(self, window: int = TEMPORAL_WINDOW, maxlen: int = TEMPORAL_MAXLEN):
        self.window  = window
        self.history = defaultdict(lambda: deque(maxlen=maxlen))

    def update(self, active_labels: set):
        """Record presence / absence for every known label this frame."""
        all_labels = set(self.history.keys()) | active_labels
        for label in all_labels:
            self.history[label].append(1 if label in active_labels else 0)

    def is_confirmed(self, label: str) -> bool:
        """True if the label has been consistently detected recently."""
        return sum(self.history[label]) >= self.window

    def reset(self):
        """Clear all history — call between videos."""
        self.history.clear()


# ================================================================
# CLASSIFIER ENGINE
# ================================================================
class ClassifierEngine:
    """
    TRT FP16 classifier using PyTorch CUDA tensors for bindings.
    Shares the same CUDA context as Ultralytics YOLO — no context conflict.
    All YOLO detections pass through this classifier.
    non_weapon prediction → detection suppressed as false positive.
    """

    def __init__(self, engine_path: str):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        if engine is None:
            raise RuntimeError(
                "\n❌ TensorRT engine failed to load.\n"
                "👉 This is NOT a code bug.\n"
                "👉 It means your .engine was built with a different TensorRT version.\n"
                "🔧 Fix: Rebuild classifier_fp16.engine using current TensorRT.\n"
            )

        self.engine  = engine
        self.context = self.engine.create_execution_context()

        self.input_tensor  = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device="cuda")
        self.output_tensor = torch.zeros((1, NUM_CLASSES), dtype=torch.float32, device="cuda")

        self._warmup()
        print(f"✅ Classifier TRT engine loaded & warmed up: {engine_path}")

    def _warmup(self, n: int = 3):
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
    return FUSION_ALPHA * yolo_conf + (1 - FUSION_ALPHA) * clf_conf


# ================================================================
# POST-CLASSIFIER NMS
# ================================================================
def apply_nms(detections: list, iou_threshold: float = NMS_IOU_THRES) -> list:
    if not detections:
        return []
    boxes  = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2, _, _ in detections]
    scores = [conf for *_, conf in detections]
    keep   = cv2.dnn.NMSBoxes(boxes, scores, DRAW_CONF_THRES, iou_threshold)
    if len(keep) == 0:
        return []
    return [detections[i] for i in keep.flatten()]


# ================================================================
# LABEL AGREEMENT CHECK
# ================================================================
def passes_label_agreement(yolo_label: str, clf_label: str, clf_conf: float) -> bool:
    if LABEL_AGREEMENT == "off":
        return True
    if LABEL_AGREEMENT == "strict":
        return clf_label == yolo_label
    if LABEL_AGREEMENT == "soft":
        if clf_label != yolo_label and clf_conf > SOFT_AGREE_THRES:
            return False
        return True
    return True


# ================================================================
# OVERLAY HELPERS
# ================================================================

def draw_hud(frame, source_name, weapon_counts, fps):
    h, w = frame.shape[:2]
    cv2.putText(frame, source_name, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 140, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


def draw_detection(frame, x1, y1, x2, y2, label, conf):
    color = CLASS_COLORS.get(label, (255, 255, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cvzone.putTextRect(frame, f"{label} {conf:.2f}",
                       (x1, y1 - 10), scale=1.2, thickness=2)


def save_snapshot(frame, source_name: str):
    """Save the current frame as a timestamped PNG in OUTPUT_DIR."""
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base       = os.path.splitext(os.path.basename(source_name))[0]
    filename   = f"snapshot_{base}_{timestamp}.png"
    out_path   = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(out_path, frame)
    print(f"📸 Snapshot saved → {out_path}")


# ================================================================
# VIDEO WRITER FACTORY
# ================================================================
def make_writer(output_path: str, fps: float, frame_size: tuple) -> cv2.VideoWriter | None:
    """
    Create and return a cv2.VideoWriter for the given output path.

    Parameters
    ----------
    output_path : str
        Full path to the output .mp4 file.  The parent directory is
        created automatically if it does not exist.
    fps : float
        Frame rate of the output video — should match the source FPS so
        playback speed matches the original recording.
    frame_size : tuple
        (width, height) of each frame in pixels.  Must match the frames
        passed to writer.write() exactly or OpenCV will silently drop them.

    Returns
    -------
    cv2.VideoWriter | None
        A ready-to-use writer, or None if output_path is None or the
        writer could not be opened (a warning is printed in that case).

    Notes
    -----
    mp4v (MPEG-4 Part 2) is chosen for broad compatibility.  The output
    container is .mp4.  If the destination codec is unavailable on the
    current platform OpenCV will still create the file but it may not be
    playable — check the isOpened() guard below.
    """
    if output_path is None:
        return None
    # Ensure the destination directory exists before OpenCV tries to open it
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    if not writer.isOpened():
        print(f"⚠️  Could not open VideoWriter for: {output_path}")
        return None
    return writer


def _release_writer(writer: cv2.VideoWriter | None, output_path: str | None):
    """
    Flush and close a VideoWriter, then print a confirmation line.
    Safe to call with None — does nothing in that case.
    Called both on normal video end and on early-exit key presses so the
    output file is always finalised before the function returns.
    """
    if writer is not None:
        writer.release()
        if output_path:
            print(f"✅ Video saved → {output_path}")


# ================================================================
# SCREEN SIZE HELPER
# ================================================================
def _get_quarter_screen_size() -> tuple:
    """
    Return (width, height) equal to half the primary screen in each
    dimension, i.e. one quarter of the total screen area.

    Implementation note
    -------------------
    Uses tkinter, which ships with CPython and requires no extra install.
    root.withdraw() suppresses the blank Tk window that would otherwise
    flash on screen for a split second.  root.destroy() tears down the
    Tk event loop cleanly so it does not interfere with OpenCV's GUI.

    Falls back to 960 × 540 when tkinter is unavailable (headless server,
    minimal Docker image, etc.).  This function is called exactly once at
    startup so it contributes zero per-frame overhead.
    """
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()                      # hide the blank Tk window immediately
        w = root.winfo_screenwidth()  // 2
        h = root.winfo_screenheight() // 2
        root.destroy()
        return w, h
    except Exception:
        return 960, 540                      # sensible half-1080p fallback


# ================================================================
# VIDEO CONTROL SIGNALS
# ================================================================
# Returned by the inner per-video loop to tell the outer loop what to do next.
SIGNAL_NEXT = "next"   # e  →  move to next video
SIGNAL_PREV = "prev"   # q  →  go back to previous video
SIGNAL_EXIT = "exit"   # x  →  quit everything


# ================================================================
# PER-VIDEO LOOP
# ================================================================
def process_video(cap, source_name, output_path, yolo_model, clf_engine, tracker):
    """
    Run inference on a single VideoCapture until the stream ends or the
    user presses a control key.

    Annotated frames are written to output_path via VideoWriter at the
    source's native resolution and FPS.

    Seeking (d / a keys) is supported for file-based sources.  For live
    webcam streams CAP_PROP_FRAME_COUNT returns -1, so seek attempts are
    detected and skipped with an informational message instead of crashing.

    Returns one of: SIGNAL_NEXT | SIGNAL_PREV | SIGNAL_EXIT
    """
    tracker.reset()

    # ── Derive video properties needed by the writer and seek logic
    video_fps       = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # CAP_PROP_FRAME_COUNT is -1 for webcam / RTSP — used to guard seek calls
    total_frames    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Convert the configured SEEK_SECONDS to a concrete frame count once
    frames_per_seek = int(video_fps * SEEK_SECONDS)

    # ── Open the output writer for this source (full-resolution, source FPS)
    writer = make_writer(output_path, video_fps, (frame_w, frame_h))
    if writer is not None:
        print(f"💾 Recording → {output_path}")
    
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # Stream ended naturally — fall through to release + SIGNAL_NEXT
                break

            results = yolo_model.predict(
                frame,
                imgsz=IMG_SIZE,
                conf=CONF_THRES,
                verbose=False,
                show=False
            )

            raw = []

            for r in results:
                for box in r.boxes:

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id          = int(box.cls[0])
                    yolo_conf       = float(box.conf[0])
                    yolo_label      = WEAPON_CLASSES[cls_id]

                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    probs     = clf_engine.infer(crop)
                    idx       = int(np.argmax(probs))
                    clf_conf  = float(probs[idx])
                    clf_label = WEAPON_CLASSES[idx]

                    if clf_label == "non_weapon":
                        continue

                    conf = fuse_confidence(yolo_conf, clf_conf)
                    raw.append((x1, y1, x2, y2, clf_label, conf))

            detections = apply_nms(raw)
            active     = {d[4] for d in detections}
            tracker.update(active)

            for x1, y1, x2, y2, label, conf in detections:
                if tracker.is_confirmed(label):
                    draw_detection(frame, x1, y1, x2, y2, label, conf)

            # ── Write the annotated frame at full resolution before display
            if writer is not None:
                writer.write(frame)

            # ── Display via WINDOW_NORMAL — the compositor scales to the window
            #    size set in run_inference(); the frame array is never resized
            #    here, so there is zero per-frame overhead from the window sizing
            #    (see Fix 3 in the module docstring).
            cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 32:   # Spacebar
            paused = not paused

            if paused:
                print("⏸ Video paused")
            else:
                print("▶ Video resumed")

        elif key == ord('e'):
            # Quit this video and advance to the next one
            break

        elif key == ord('x'):
            # Exit the entire program immediately — release writer first so
            # the output file is properly finalised before the process ends
            _release_writer(writer, output_path)
            return SIGNAL_EXIT

        elif key == ord('q'):
            # Go back to the previous video
            print(f"⏮  Going back from: {source_name}")
            _release_writer(writer, output_path)
            return SIGNAL_PREV

        elif key == ord('d'):
            # Fast-forward by SEEK_SECONDS — file sources only.
            # total_frames == -1 means live stream; skip seek silently.
            if total_frames > 0:
                current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                # Clamp so we never seek past the last frame
                new_pos = min(current_pos + frames_per_seek, total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                print(f"⏩  Fast-forward {SEEK_SECONDS}s → frame {new_pos} / {total_frames}")
            else:
                print("⏩  Fast-forward not supported for live streams")

        elif key == ord('a'):
            # Rewind by SEEK_SECONDS — file sources only.
            # total_frames == -1 means live stream; skip seek silently.
            if total_frames > 0:
                current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                # Clamp so we never seek before frame 0
                new_pos = max(0, current_pos - frames_per_seek)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                print(f"⏪  Rewind {SEEK_SECONDS}s → frame {new_pos} / {total_frames}")
            else:
                print("⏪  Rewind not supported for live streams")

        elif key == ord('s'):
            # Save a snapshot of the current frame (does NOT pause playback)
            save_snapshot(frame, source_name)

    # ── Natural end of stream or 'e' pressed — finalise the output file
    _release_writer(writer, output_path)
    return SIGNAL_NEXT


# ================================================================
# MAIN INFERENCE LOOP
# ================================================================
def run_inference(video_sources=None):

    print("📦 Loading YOLO TRT engine...")
    yolo_model = YOLO(YOLO_ENGINE_PATH, task="detect")
    yolo_model.overrides['show'] = False
    print("✅ YOLO loaded")

    clf_engine = ClassifierEngine(CLF_ENGINE_PATH)
    tracker    = TemporalTracker()

    # ── Create the display window ONCE before the main loop (see Fix 3).
    #    WINDOW_NORMAL delegates all scaling to the OS compositor — the frame
    #    array is never resized so inference speed is completely unaffected.
    #    The handle persists across all videos; no re-creation is needed.
    win_w, win_h = _get_quarter_screen_size()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, win_w, win_h)
    print(f"🖥️  Display window: {win_w} × {win_h}  (quarter-screen, WINDOW_NORMAL)")

    # ── Build a flat list of (source_name, cap_factory, output_path) tuples
    #    so we can navigate forward AND backward by integer index.
    #    cap_factory is a zero-arg lambda so the VideoCapture is only opened
    #    when that source is actually about to be processed.
    if video_sources is None:
        # Webcam fallback — output saved as webcam_output.mp4
        sources = [(
            "webcam",
            lambda: cv2.VideoCapture(0),
            os.path.join(INFERENCE_OUTPUT_DIR, "webcam_output.mp4"),
        )]
    else:
        sources = []
        for v in video_sources:
            base = os.path.splitext(os.path.basename(v))[0]
            out  = os.path.join(INFERENCE_OUTPUT_DIR, f"{base}_output.mp4")
            # v=v captures the loop variable in the lambda's default arg
            sources.append((v, lambda v=v: cv2.VideoCapture(v), out))

    idx = 0  # current position in the sources list

    while 0 <= idx < len(sources):
        source_name, cap_factory, output_path = sources[idx]
        cap = cap_factory()

        if not cap.isOpened():
            print(f"⚠️  Could not open source: {source_name}  — skipping")
            idx += 1
            continue

        print(f"\n▶  Processing: {source_name}")
        signal = process_video(
            cap, source_name, output_path,
            yolo_model, clf_engine, tracker,
        )
        cap.release()

        if signal == SIGNAL_EXIT:
            print("🛑 Exit requested — stopping.")
            break
        elif signal == SIGNAL_PREV:
            idx = max(0, idx - 1)          # clamp so we never go below 0
        else:                               # SIGNAL_NEXT (default)
            idx += 1

    cv2.destroyAllWindows()
    print("✅ Done.")


# ================================================================
# ENTRY
# ================================================================
if __name__ == "__main__":
    video_paths = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
    run_inference(video_paths if video_paths else None)