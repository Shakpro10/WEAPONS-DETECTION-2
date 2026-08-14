import os
from ultralytics import YOLO
import glob
import cv2
import math
import torch
import random
import numpy as np
from config import BEST_WEIGHTS, RANDOM_SEED, IMG_SIZE, DEVICE

# -----------------------------
# Class definitions with colors
# -----------------------------
CLASSES = {
    0: {"name": "Handgun",                  "color": (128,   0, 128)},  # Purple
    1: {"name": "Heavy Weapon",             "color": (255,   0,   0)},  # Blue
    2: {"name": "Knife",                    "color": (  0, 255,   0)},  # Green
    3: {"name": "Rifle",                    "color": (  0,   0, 255)},  # Red
    4: {"name": "Rocket_Grenade Launcher",  "color": (  0, 165, 255)},  # Orange
}

# ----------------------------
# Reproducibility
# ----------------------------
def set_seeds(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =========================
# Load YOLO model
# =========================
best_pt = BEST_WEIGHTS

device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
model  = YOLO(best_pt).to(device)

# FIX 1: Warm-up pass so the first real frame isn't sluggish
# (YOLO/PyTorch compiles CUDA kernels on the first forward pass;
#  running a dummy frame here absorbs that one-time cost.)
_dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
model(_dummy, imgsz=IMG_SIZE, verbose=False)
del _dummy

# FIX 2: Export half-precision (FP16) for GPU — roughly 2× faster inference
# at no meaningful accuracy cost for detection. No-op on CPU.
USE_HALF = torch.cuda.is_available()

WINDOW_NAME = "YOLOv26 Weapon Detection (One-to-One Head)"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.moveWindow(WINDOW_NAME, 200, 80)

exit_all   = False
frame_step = 30   # frames to skip per fast-forward / rewind keypress


# =========================
# Run detection on a video
# =========================
def run_video_inference(video_source=None, conf_thres=0.25):
    """Run detection on a video using YOLO one-to-one head (no NMS required)."""
    global exit_all
    set_seeds()

    # ── Build list of capture sources ────────────────────────────────
    if video_source is None:
        print("  Starting webcam stream …")
        sources      = [cv2.VideoCapture(0)]
        source_names = ["Webcam"]
    elif isinstance(video_source, (list, tuple)):
        sources      = [cv2.VideoCapture(vp) for vp in video_source]
        source_names = [os.path.basename(vp)  for vp in video_source]
    elif isinstance(video_source, str) and os.path.exists(video_source):
        sources      = [cv2.VideoCapture(video_source)]
        source_names = [os.path.basename(video_source)]
    else:
        print("  Error: Invalid video source(s).")
        return

    # ── Process each video ────────────────────────────────────────────
    for cap, video_name in zip(sources, source_names):
        if not cap.isOpened():
            print(f"  Error: Could not open — {video_name}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS)
        frame_index  = 0
        seek_pending = False  # FIX 3: only seek on A/D, not every frame

        print("=" * 60)
        print(f"  Processing : {video_name}")
        print(f"  Frames     : {total_frames}  |  FPS: {fps:.2f}")
        print("=" * 60)

        while True:
            # FIX 3: seek only when A/D was pressed
            if seek_pending:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                seek_pending = False

            success, frame = cap.read()
            if not success:
                break  # end of video

            frame_index += 1

            # ── Run inference ─────────────────────────────────────────
            # FIX 4: half=USE_HALF enables FP16 on CUDA for ~2× speed-up.
            # FIX 5: removed redundant inner `if conf > 0.4` gate — conf_thres
            #         already filters at the model level; the double-check
            #         wastes CPU and causes detections between 0.25–0.40 to be
            #         drawn but then silently discarded, creating confusing behavior.
            results = model(
                frame,
                conf=conf_thres,
                imgsz=IMG_SIZE,
                max_det=100,
                stream=True,
                verbose=False,
                half=USE_HALF,
                end2end=True,  # one-to-one head — no NMS required
                show=False
            )

            for r in results:
                boxes = r.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                # FIX 6: pull all box data to CPU numpy in one batch call
                # instead of indexing each tensor one-by-one inside the loop.
                xyxy_arr  = boxes.xyxy.cpu().numpy().astype(int)   # (N,4)
                conf_arr  = boxes.conf.cpu().numpy()               # (N,)
                cls_arr   = boxes.cls.cpu().numpy().astype(int)    # (N,)

                for (x1, y1, x2, y2), conf, cls in zip(xyxy_arr, conf_arr, cls_arr):
                    class_info   = CLASSES.get(cls, {"name": f"Class_{cls}", "color": (255, 255, 255)})
                    currentClass = class_info["name"]
                    myColor      = class_info["color"]

                    label = f"{currentClass} {conf:.2f}"

                    # FIX 7: replaced cvzone.putTextRect() with plain OpenCV calls.
                    # cvzone wraps cv2 internally but adds Python overhead on every
                    # call; at 30+ fps this adds up.  Two cv2 calls do the same job
                    # faster and without the extra dependency.
                    cv2.rectangle(frame, (x1, y1), (x2, y2), myColor, 2)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), myColor, -1)
                    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # ── Overlay video info ────────────────────────────────────
            info_text = f"Video: {video_name}  |  Frame: {frame_index}/{total_frames}"
            fps_text  = f"FPS: {fps:.1f}  |  One-to-One Head (No NMS)  |  Conf: {conf_thres}"
            cv2.putText(frame, info_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255,   0), 2)
            cv2.putText(frame, fps_text,  (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (  0, 255, 255), 2)

            cv2.imshow(WINDOW_NAME, frame)

            # FIX 8: waitKey(1) instead of fps-based delay.
            # The original int(1000/fps) block (e.g. 33 ms at 30 fps) added
            # artificial sleep ON TOP of inference time, doubling perceived lag.
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("  Skipping to next video …")
                break
            elif key == ord('e'):
                print("  Exiting all videos …")
                exit_all = True
                break
            elif key == ord('d'):  # fast-forward
                frame_index  = min(frame_index + frame_step, total_frames - 1)
                seek_pending = True
            elif key == ord('a'):  # rewind
                frame_index  = max(frame_index - frame_step, 0)
                seek_pending = True

        cap.release()
        print(f"  Finished: {video_name}")

        if exit_all:
            break

    cv2.destroyAllWindows()
    print("  All videos processed.")


# ===============================
# Main
# ===============================
if __name__ == "__main__":
    print("=" * 60)
    print("  YOLOv26 Weapon Detection — Video Inference")
    print("=" * 60)
    print("  Controls:")
    print("    Q — Skip to next video")
    print("    E — Exit all videos")
    print("    D — Fast forward 30 frames")
    print("    A — Rewind 30 frames")
    print("=" * 60)

    video_dir = "./test_videos"

    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv',
                        '*.MP4', '*.AVI', '*.MOV', '*.MKV']
    video_paths = []
    for ext in video_extensions:
        video_paths.extend(glob.glob(os.path.join(video_dir, ext)))

    if not video_paths:
        print(f"  No video files found in: {video_dir}")
        print("  Starting webcam instead …")
        run_video_inference(None, conf_thres=0.60)
    else:
        print(f"  Found {len(video_paths)} video(s):")
        for i, vp in enumerate(video_paths, 1):
            print(f"    {i}. {os.path.basename(vp)}")
        print("=" * 60)
        run_video_inference(video_paths, conf_thres=0.60)

    print("=" * 60)
    print("  Video inference completed.")
    print("=" * 60)
