# YOLOv26 Weapon Detection

**Real-time multi-class weapon detection for video surveillance using YOLOv26, a MobileNetV3-Small verification classifier, PyTorch, OpenCV, and NVIDIA TensorRT FP16.**

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-ee4c2c)
![YOLOv26](https://img.shields.io/badge/Detector-YOLOv26-green)
![TensorRT](https://img.shields.io/badge/TensorRT-FP16-76B900)
![Classes](https://img.shields.io/badge/Weapon_Classes-5-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Overview

This repository implements a **two-stage weapon detection and verification pipeline** designed for real-time or near-real-time analysis of recorded video and webcam streams.

The system combines:

1. **YOLOv26s object detection** for locating candidate weapons.
2. **MobileNetV3-Small classification** for verifying the detected crop and rejecting false positives.
3. **TensorRT FP16 engines** for deployment-oriented GPU inference.
4. **YOLO tracking with ByteTrack** for temporal object identity.
5. **Per-track classifier smoothing** to reduce frame-to-frame label flicker.
6. **Confidence fusion, label-agreement checks, crop-quality filtering, and post-classifier NMS** to improve final detections.

The detector is trained at **640 × 640** resolution. The verification classifier operates on **224 × 224** crops.

The project is intended as a practical computer-vision pipeline for weapon-detection research, prototyping, and authorized security applications. It should be validated against representative target footage before operational deployment.

---

## Detection Classes

The detector recognizes five weapon classes:

|  ID | Class                   |
| --: | ----------------------- |
|   0 | Handgun                 |
|   1 | Heavy Weapon            |
|   2 | Knife                   |
|   3 | Rifle                   |
|   4 | Rocket_Grenade Launcher |

The classifier uses the same five detector-aligned weapon indices and adds a sixth class:

|  ID | Class                   | Role                         |
| --: | ----------------------- | ---------------------------- |
|   0 | Handgun                 | Weapon                       |
|   1 | Heavy Weapon            | Weapon                       |
|   2 | Knife                   | Weapon                       |
|   3 | Rifle                   | Weapon                       |
|   4 | Rocket_Grenade Launcher | Weapon                       |
|   5 | `non_weapon`            | Reject / false-positive gate |

Keeping indices `0–4` identical between the detector and classifier is critical. The classifier explicitly overrides the default alphabetical `ImageFolder` class ordering so that `non_weapon` remains index `5`.

---

## System Architecture

```text
                         INPUT FRAME / VIDEO
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │ YOLOv26s Detector   │
                       │ 640 × 640           │
                       │ TensorRT FP16       │
                       └──────────┬──────────┘
                                  │
                         Candidate bounding boxes
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │ Crop Quality Filter │
                       │ Min area = 1500 px² │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │ MobileNetV3-Small   │
                       │ 224 × 224           │
                       │ TensorRT FP16       │
                       └──────────┬──────────┘
                                  │
                         Class + probability
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
             non_weapon                  Weapon class
                    │                           │
                    ▼                           ▼
                REJECT                 Confidence / agreement
                                                │
                                                ▼
                                      Confidence fusion
                                                │
                                                ▼
                                      YOLO object tracking
                                           ByteTrack
                                                │
                                                ▼
                                      Per-track smoothing
                                                │
                                                ▼
                                        Temporal confirmation
                                                │
                                                ▼
                                       Post-classifier NMS
                                                │
                                                ▼
                                      FINAL WEAPON DETECTION
```

### Why two models?

The detector is responsible for **where** a possible weapon is. The classifier then evaluates the extracted crop and acts as a second-stage verification gate.

A classifier prediction of `non_weapon` suppresses the candidate. Low classifier confidence also suppresses the candidate. This design makes the classifier a false-positive guard rather than simply a second independent classifier.

---

## Model Components

### 1. YOLOv26s Detector

The detector is trained with the Ultralytics YOLO interface using pretrained `yolo26s.pt` weights.

Configured detector settings include:

| Parameter               |               Value |
| ----------------------- | ------------------: |
| Detector                |            YOLOv26s |
| Input size              |           640 × 640 |
| Epochs                  |                 300 |
| Batch size              |                  16 |
| Initial LR (`lr0`)      |                0.01 |
| Final LR (`lrf`)        |                0.01 |
| Early stopping patience |                  15 |
| Optimizer               |               MuSGD |
| Warmup                  |            3 epochs |
| Cosine LR               |             Enabled |
| AMP                     |             Enabled |
| Workers                 |                   8 |
| Random seed             |                  42 |
| Device                  | CUDA when available |

The training script also enables augmentation including HSV variation, rotation, translation, scaling, shear, perspective transformation, copy-paste, horizontal flipping, mosaic, erasing, and closing mosaic augmentation during the final training period.

### 2. MobileNetV3-Small Verification Classifier

The second-stage classifier is a pretrained **MobileNetV3-Small** whose final classification layer is replaced with six outputs.

Input preprocessing:

- Resize to `224 × 224`
- Convert to tensor
- ImageNet normalization:
  - Mean: `[0.485, 0.456, 0.406]`
  - Standard deviation: `[0.229, 0.224, 0.225]`

Training configuration:

| Parameter                         |                          Value |
| --------------------------------- | -----------------------------: |
| Architecture                      |              MobileNetV3-Small |
| Classes                           |                              6 |
| Input size                        |                      224 × 224 |
| Batch size                        |                             32 |
| Maximum epochs                    |                             50 |
| Initial LR                        |                         `1e-4` |
| Weight decay                      |                         `1e-4` |
| Optimizer                         |                          AdamW |
| LR scheduler                      |              ReduceLROnPlateau |
| Frozen-backbone phase             |                       5 epochs |
| Full-backbone LR after unfreezing |                         `1e-5` |
| Early stopping patience           |                       7 epochs |
| Selection metric                  | Macro F0.5 over weapon classes |

The classifier uses weighted cross-entropy to compensate for class imbalance.

The reported training counts in `classifier_train.py` are:

| Class                   | Training samples |
| ----------------------- | ---------------: |
| Handgun                 |            4,642 |
| Heavy Weapon            |            2,012 |
| Knife                   |            7,362 |
| Rifle                   |            9,226 |
| Rocket_Grenade Launcher |            1,600 |
| `non_weapon`            |           10,978 |

The classifier deliberately gives the `non_weapon` class a dedicated index because a detector crop can contain a visually weapon-like region that is not actually a weapon.

---

## Dataset Pipeline

The project uses a YOLO-format detector dataset containing:

```text
dataset/
├── detector_dataset/
│    └── Weapons_Detection_New_v1_version_1_yolo26/
│        ├── train/
│        │   ├── images/
│        │   └── labels/
│        ├── valid/
│        │   ├── images/
│        │   └── labels/
│        ├── test/
│        │   ├── images/
│        │   └── labels/
│        └── data.yaml
│
├── classifier_dataset/
│    ├── train/
│    │    ├── Handgun/
│    │    ├── Heavy Weapon/
│    │    ├── Knife/
│    │    ├── Rifle/
│    │    └──  Rocket_Grenade Launcher/
│    │
│    ├── valid/
│    │    ├── Handgun/
│    │    ├── Heavy Weapon/
│    │    ├── Knife/
│    │    ├── Rifle/
│    │    └──  Rocket_Grenade Launcher/
│    │
│    └── test/
│        ├── Handgun/
│        ├── Heavy Weapon/
│        ├── Knife/
│        ├── Rifle/
│        └── Rocket_Grenade Launcher/
│
└── hard_neg_mine/

```

The detector classes are defined in `config.py`.

### Generating the classifier dataset

`bbox_region_gen.py` converts detector annotations into an `ImageFolder`-style classification dataset.

For each labeled detector object:

1. The YOLO normalized bounding box is converted to pixel coordinates.
2. A **15% padding** is applied around the box.
3. The padded crop is saved under its weapon class.
4. Images with missing or empty label files are treated as negative images and copied into `non_weapon`.

Output:

```text
dataset/
└── classifier_dataset/
    ├── train/
    │   ├── Handgun/
    │   ├── Heavy Weapon/
    │   ├── Knife/
    │   ├── Rifle/
    │   ├── Rocket_Grenade Launcher/
    │   └── non_weapon/
    ├── valid/
    │   └── ...
    └── test/
        └── ...
```

Run:

```bash
python bbox_region_gen.py
```

### Hard-negative augmentation

`hard_neg_mine.py` creates additional augmented negative examples from the existing `non_weapon` images.

The augmentation choices include:

- Horizontal flip
- Rotation
- Gaussian blur
- HSV variation
- Translation

The script generates approximately **80% of the available negative-image count** as additional samples in:

```text
dataset/classifier_dataset/train/non_weapon_aug/
```

If these generated samples are used for classifier training, they should be incorporated into the expected `non_weapon` training class directory because `classifier_train.py` expects the six exact class folders listed above.

---

## Training the Detector

Configure the dataset and training parameters in `config.py`.

Important settings include:

```python
IMG_SIZE = 640
EPOCHS = 300
BATCH_SIZE = 16
LR_INITIAL = 0.01
LR_FINAL = 0.01
PATIENCE = 15
RANDOM_SEED = 42
```

Then run:

```bash
python detector_train.py
```

The training pipeline:

1. Sets Python, NumPy, and PyTorch random seeds.
2. Loads `yolo26s.pt`.
3. Trains on the configured YOLO dataset.
4. Uses CUDA when available.
5. Enables AMP.
6. Saves training plots and checkpoints.
7. Reads `results.csv` after training.
8. Selects the epoch with the highest `mAP50-95`.
9. Writes the best-epoch summary to `runs/detect/train/train_metadata.json`.

The best and last detector weights are saved as:

```text
runs/detect/train/weights/best.pt
runs/detect/train/weights/last.pt
```

---

## Recorded Detector Performance

The uploaded `train_metadata.json` records the following best-epoch detector result:

| Metric       | Recorded value |
| ------------ | -------------: |
| Best epoch   |        **258** |
| Precision    |     **95.12%** |
| Recall       |     **92.45%** |
| F1 score     |     **93.76%** |
| mAP@0.5      |     **95.24%** |
| mAP@0.5:0.95 |     **80.37%** |

The training script selects the best epoch according to `metrics/mAP50-95(B)`.

> **Evaluation note:** These are the metrics recorded in the supplied training metadata. The available artifact does not identify them as a held-out real-world deployment benchmark. They should therefore be treated as recorded model-training/validation results rather than a guarantee of performance on unseen surveillance footage.

---

## Training Artifacts

A typical YOLO training run produces:

```text
runs/
└── detect/
    └── train/
        ├── weights/
        │   ├── best.pt
        │   └── last.pt
        ├── args.yaml
        ├── results.csv
        ├── results.png
        ├── confusion_matrix.png
        ├── confusion_matrix_normalized.png
        ├── BoxF1_curve.png
        ├── BoxP_curve.png
        ├── BoxPR_curve.png
        ├── BoxR_curve.png
        ├── labels.jpg
        ├── train_batch0.jpg
        ├── train_batch1.jpg
        ├── train_batch2.jpg
        ├── val_batch0_labels.jpg
        ├── val_batch0_pred.jpg
        ├── val_batch1_labels.jpg
        ├── val_batch1_pred.jpg
        ├── val_batch2_labels.jpg
        ├── val_batch2_pred.jpg
        └── train_metadata.json
```

The repository screenshots also show a dedicated `runs/detect/train` artifact directory and separate verification reports.

---

## Training the Classifier

First generate the classifier crops:

```bash
python bbox_region_gen.py
```

Optionally generate additional negative examples:

```bash
python hard_neg_mine.py
```

Then train MobileNetV3-Small:

```bash
python classifier_train.py
```

The classifier saves its best state dictionary to:

```text
classifier_weights/best_model.pth
```

The best model is selected using **macro F0.5 across the five weapon classes**, where `beta=0.5` weights precision more heavily than recall.

The classifier also reports a dedicated reject-gate precision and recall for the `non_weapon` class.

No final classifier accuracy/F0.5 value is claimed in this README because the supplied project artifacts contain the verification script but do not provide a corresponding completed `classifier_verification.json` result with final numeric values.

---

## TensorRT Deployment

The project contains separate export paths for the detector and classifier.

### YOLO detector → TensorRT

`detector_engine_export.py` exports the trained YOLO detector directly to a TensorRT engine through Ultralytics:

```bash
python detector_engine_export.py
```

The configured export uses:

- `imgsz=640`
- `batch=1`
- `half=True`
- TensorRT workspace of 8 GB

The resulting engine is expected alongside the detector weights as:

```text
runs/detect/train/weights/best.engine
```

A quick inference test is also performed after export.

### MobileNet classifier → ONNX → TensorRT

`class_onnx_trt_conv.py` performs:

```text
best_model.pth
      │
      ▼
PyTorch MobileNetV3-Small
      │
      ▼
ONNX (opset 18)
      │
      ▼
Cleaned ONNX without external weight dependency
      │
      ▼
TensorRT FP16 engine
```

Run:

```bash
python class_onnx_trt_conv.py
```

The configured outputs are:

```text
classifier_weights/
├── best_model.pth
├── classifier.onnx
└── classifier_fp16.engine
```

The classifier engine uses an input shape of:

```text
1 × 3 × 224 × 224
```

and six output classes.

> **Deployment note:** TensorRT engines are deployment artifacts rather than portable source models. Build and validate the engine in the target CUDA/TensorRT environment rather than assuming an engine built on one machine will be interchangeable with every other GPU/software stack.

---

## FP16 Verification

The repository includes dedicated scripts for comparing the original PyTorch model with the TensorRT FP16 engine.

### Detector verification

Run:

```bash
python detector_fp16_model_verifiction.py
```

The script compares:

- PyTorch `.pt` FP32
- TensorRT `.engine` FP16

using the same validation configuration:

| Parameter              |    Value |
| ---------------------- | -------: |
| Image size             |      640 |
| Batch size             |        1 |
| Rectangular validation | Disabled |
| Device                 |    GPU 0 |

It records:

- mAP@0.5
- mAP@0.5:0.95
- Per-class metrics
- FP16 drop
- Production-safety verdict

Output:

```text
verification_reports/yolo_verification.json
```

The script classifies the observed mAP@0.5 drop as:

- `≤ 1%` — safe for production
- `> 1% and ≤ 2%` — acceptable but monitor
- `> 2%` — consider the FP32 engine or further model work

### Classifier verification

Run:

```bash
python classifier_fp16_pt_verifiction.py
```

It compares:

- MobileNetV3-Small PyTorch FP32
- MobileNetV3-Small TensorRT FP16

using the classifier test dataset.

Output:

```text
verification_reports/classifier_verification.json
```

The report contains overall accuracy, per-class accuracy, FP32-vs-FP16 drop, and a verdict.

---

## Production Inference Pipeline

The main deployment-oriented inference implementation is:

```text
weapon_detection_v3.py
```

Run:

```bash
python weapon_detection_v3.py
```

The script searches `./test_videos` for:

```text
.mp4
.avi
.mov
.mkv
```

If no videos are found, it falls back to the default webcam:

```text
OpenCV camera index 0
```

### Configured model paths

The inference script expects:

```text
./models/detector/best.engine
./models/classifier/classifier_fp16.engine
```

The export scripts produce engines under their training/weights directories by default, so for deployment you should either:

- place/copy the built engines under `models/detector` and `models/classifier`, or
- update `YOLO_ENGINE_PATH` and `CLF_ENGINE_PATH` in `weapon_detection_v3.py`.

---

## Inference Decision Pipeline

For every YOLO candidate, the production inference script applies the following sequence:

### 1. YOLO confidence gate

```python
CONF_THRES = 0.40
```

Candidates below this threshold are not passed through the verification stage.

### 2. Crop-quality filter

```python
MIN_CROP_AREA = 1500
```

Very small candidate regions are discarded before classifier inference.

### 3. MobileNetV3 classifier

The crop is resized to `224 × 224`, ImageNet-normalized, and evaluated by the TensorRT FP16 classifier.

### 4. `non_weapon` rejection

If the classifier predicts:

```text
non_weapon
```

the candidate is suppressed.

### 5. Classifier confidence gate

```python
CLF_CONF_THRES = 0.55
```

Low-confidence classifier predictions are rejected.

### 6. YOLO/classifier label agreement

The configured mode is:

```python
LABEL_AGREEMENT = "soft"
SOFT_AGREE_THRES = 0.70
```

In soft mode, a classifier prediction is allowed to differ from the YOLO label unless the classifier is sufficiently confident in the different class.

### 7. Confidence fusion

Final confidence is calculated as:

```text
0.45 × YOLO confidence
+
0.55 × classifier confidence
```

because:

```python
FUSION_ALPHA = 0.45
```

### 8. Track-level classifier smoothing

The classifier maintains a rolling probability history of:

```python
CLF_SMOOTH_LEN = 5
```

probability vectors per tracked object.

### 9. Temporal confirmation

A track must be confirmed for at least:

```python
TRACK_CONFIRM_FRAMES = 3
```

frames before it is drawn as a final detection.

### 10. Stale-track cleanup

Track state is removed after:

```python
STALE_TRACK_FRAMES = 30
```

frames without being observed.

### 11. Post-classifier NMS

Final detections are filtered using:

```python
NMS_IOU_THRES = 0.40
```

### 12. Final visualization

Accepted detections display:

```text
[Track ID] Class  Confidence
```

The HUD also displays the source name, live FPS, and active per-class detection counts.

---

## Inference Configuration

The principal production inference settings are:

| Parameter                    |     Value |
| ---------------------------- | --------: |
| Detector input               | 640 × 640 |
| YOLO confidence              |      0.40 |
| Classifier confidence        |      0.55 |
| Final draw confidence        |      0.55 |
| NMS IoU                      |      0.40 |
| YOLO confidence weight       |      0.45 |
| Classifier confidence weight |      0.55 |
| Minimum crop area            | 1,500 px² |
| Classifier smoothing         | 5 samples |
| Track confirmation           |  3 frames |
| Stale-track timeout          | 30 frames |
| Label agreement              |      Soft |
| Soft agreement threshold     |      0.70 |
| Tracker                      | ByteTrack |
| Classifier frame skip        |         1 |

`CLF_FRAME_SKIP = 1` means classifier inference is performed every frame. Increasing this value can reduce classifier workload on weaker GPUs by reusing cached classifier results between inference frames.

---

## Video Controls

The inference window supports:

| Key | Action                                          |
| --- | ----------------------------------------------- |
| `Q` | Stop the current video and continue to the next |
| `E` | Exit all videos                                 |
| `D` | Seek forward 30 frames                          |
| `A` | Seek backward 30 frames                         |
| `S` | Save a screenshot                               |

Processed videos are written to:

```text
output_videos_1/
```

when `OUTPUT_DIR` is enabled.

The output writer uses OpenCV's `mp4v` codec.

---

## Repository Structure

A production repository can be organized as:

```text
YOLOv26-Weapon-Detection/
│
├── config.py
│
├── detector_train.py
├── detector_engine_export.py
├── detector_fp16_model_verifiction.py
├── detector_test_metrics.py
├── detector_test.py
├── detector_resume_train.py
├── detector_fine_tuning.py
│
├── bbox_region_gen.py
├── hard_neg_mine.py
├── classifier_train.py
├── class_onnx_trt_conv.py
├── classifier_fp16_pt_verifiction.py
├── class_onnx_trt_conv.py
├── det_class_test.py
├── pretrained_test.py
│
├── weapon_detection.py
├── weapon_detection_v3.py
│
├── dataset/
│   ├── detector_dataset/
│   └── classifier_dataset/
│
├── models/
│   ├── detector/
│   │   └── best.engine
│   └── classifier/
│       └── classifier_fp16.engine
│
├── classifier_weights/
│   ├── best_model.pth
│   ├── classifier.onnx
│   └── classifier_fp16.engine
│
├── runs/
│   └── detect/
│       └── train/
│           ├── weights/
│           │   ├── best.pt
│           │   ├── last.pt
│           │   └── best.engine
│           ├── results.csv
│           ├── results.png
│           ├── confusion_matrix.png
│           ├── confusion_matrix_normalized.png
│           └── train_metadata.json
│
├── verification_reports/
│   ├── yolo_verification.json
│   └── classifier_verification.json
│
├── test_videos/
├── output_videos_1/
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

Some filenames/directories above are shown because they appear in the supplied project tree or scripts; the repository should only retain the files actually needed for the intended workflow.

---

## Key Scripts

| Script                               | Purpose                                                                  |
| ------------------------------------ | ------------------------------------------------------------------------ |
| `config.py`                          | Central detector dataset, model, class, training, and path configuration |
| `detector_train.py`                  | Trains the YOLOv26 detector and records best-epoch metadata              |
| `detector_engine_export.py`          | Exports the YOLO detector to TensorRT FP16                               |
| `detector_fp16_model_verifiction.py` | Compares detector FP32 PyTorch and TensorRT FP16 metrics                 |
| `bbox_region_gen.py`                 | Generates classifier crops from YOLO annotations                         |
| `hard_neg_mine.py`                   | Generates additional augmented `non_weapon` samples                      |
| `classifier_train.py`                | Trains MobileNetV3-Small verification classifier                         |
| `class_onnx_trt_conv.py`             | Exports classifier to ONNX and builds TensorRT FP16 engine               |
| `classifier_fp16_pt_verifiction.py`  | Compares classifier PyTorch FP32 and TensorRT FP16 accuracy              |
| `weapon_detection_v3.py`             | Main tracking + detector + classifier production inference pipeline      |
| `det_class_test.py`                  | Earlier detector/classifier video testing implementation                 |
| `verification_reports/*.json`        | Stores FP32/FP16 verification results                                    |

---

## Reproducibility

The detector training pipeline uses:

```python
RANDOM_SEED = 42
```

The seed is applied to Python's `random`, NumPy, PyTorch, and CUDA when available.

The classifier also relies on CUDA when available and uses deterministic class ordering rather than `ImageFolder`'s default alphabetical mapping.

Exact reproducibility can still vary with GPU hardware, CUDA, PyTorch, TensorRT, driver versions, and other environment details.

---

## Requirements

The supplied scripts use the following major packages/components:

- Python
- PyTorch
- TorchVision
- Ultralytics
- TensorRT
- PyCUDA
- ONNX
- OpenCV
- NumPy
- pandas
- scikit-learn
- cvzone

A compatible **NVIDIA GPU/CUDA environment** is required for the TensorRT FP16 deployment path and GPU inference.

The repository should provide a `requirements.txt` containing the versions validated for the target environment. The uploaded source files do not provide a complete pinned dependency list, so this README intentionally does not invent package versions.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
```

### 2. Create a virtual environment

#### Windows

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install project dependencies

If the repository contains a validated requirements file:

```bash
pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build and TensorRT stack appropriate for the deployment GPU before running TensorRT export or inference.

---

## Model Artifacts and GitHub

Large binary artifacts should generally not be committed directly to a normal Git repository.

The project contains model artifacts such as:

```text
*.pt
*.pth
*.engine
*.onnx
```

If these files exceed GitHub's normal repository/file limits, keep them outside the source repository or use an appropriate artifact/model-storage solution such as Git LFS or a model registry.

A practical `.gitignore` can include:

```gitignore
# Python
__pycache__/
*.py[cod]

# Virtual environments
.venv/
venv/

# Model artifacts
*.pt
*.pth
*.engine
*.onnx
*.onnx.data

# Training outputs
runs/

# Generated inference outputs
output_videos_1/
*.mp4

# Local datasets
dataset/

# Local test media
test_videos/

# IDE
.vscode/
.idea/
```

If you want the repository to be immediately runnable by other users, publish the required model artifacts separately and document their download locations.

---

## Limitations

This system is not a universal or definitive weapon detector.

Expected failure modes include:

- Very small or distant weapons
- Severe motion blur
- Strong compression artifacts
- Heavy occlusion
- Poor illumination
- Unusual viewpoints
- Objects visually similar to weapons
- Weapon types not represented adequately in training data
- Poor-quality crops
- Dense scenes and overlapping objects
- Domain shift between training data and deployment cameras

The classifier is intentionally optimized as a false-positive rejection stage, but it can also reject true weapons when the crop is ambiguous or confidence is insufficient.

The reported detector metrics do not establish real-world performance across arbitrary CCTV environments.

---

## Safety and Responsible Use

This project is intended for **research, computer-vision development, surveillance-system prototyping, and authorized security applications**.

Model predictions should be treated as automated visual detections, not as definitive determinations.

Before using the system for consequential decisions:

- Validate it on representative footage from the target environment.
- Measure both false positives and false negatives.
- Tune thresholds against the actual operational requirements.
- Maintain appropriate human review and escalation procedures.
- Audit performance across cameras, lighting conditions, and scene types.
- Comply with applicable privacy, surveillance, data-protection, and other legal requirements.

Do not use model output as the sole basis for decisions that could materially affect a person's safety, rights, or access to services.

---

## Future Improvements

Potential next steps include:

- Quantitative end-to-end FPS benchmarking on target GPUs
- INT8 TensorRT optimization and calibration
- RTSP/IP-camera support
- Multi-camera scheduling
- GPU-accelerated video decoding
- Event/alert generation
- Camera-specific threshold calibration
- Per-class threshold optimization
- Additional hard-negative mining
- Expanded and more diverse CCTV training data
- Cross-camera and cross-environment evaluation
- Automated model/artifact versioning
- Containerized deployment
- Centralized inference logging and monitoring

---

## Acknowledgements

This project builds on open-source computer-vision and deep-learning components including:

- Ultralytics YOLO
- PyTorch
- TorchVision
- NVIDIA TensorRT
- OpenCV
- ONNX
- NumPy
- pandas
- scikit-learn
- cvzone
- PyCUDA

Third-party libraries, pretrained weights, datasets, and model components remain subject to their respective licenses and terms.

---

## License

This project is released under the **MIT License**.

Copyright (c) 2026 Shakiru Sikiru

See [`LICENSE`](LICENSE) for the full license text.

---

## Disclaimer

This repository contains software and research/deployment artifacts for automated object detection and classification.

No guarantee is made that the system will:

- Detect every weapon
- Avoid every false positive
- Correctly classify every detected object
- Maintain the recorded benchmark performance on unseen data
- Produce identical results across hardware and software environments
- Operate at a specific FPS without target-hardware benchmarking

Validate the complete pipeline—including video decoding, YOLO inference, classification, tracking, post-processing, and output rendering—on the actual hardware, cameras, and environmental conditions where it will be deployed.
