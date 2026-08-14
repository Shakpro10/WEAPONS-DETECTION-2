import os
import torch

# ==========================
# CONFIGURATION SETTINGS
# ==========================

# --- PATHS & DATASET ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"Config loaded. BASE_DIR is: {BASE_DIR}") 
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "detector_dataset", "Weapons_Detection_New_v1_version_1_yolo26")
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")

# YOLOv26 training results directory
RESULTS_DIR = "./"
BEST_WEIGHTS = os.path.join(
    RESULTS_DIR,
    "./runs/detect/train/weights",
    "best.pt"
)

LAST_WEIGHTS = os.path.join(
    RESULTS_DIR,
    "./runs/detect/train/weights",
    "last.pt"
)

# Metadata files (optional but useful)
TRAIN_METADATA_FILE = os.path.join(RESULTS_DIR, "./runs/detect/train", "train_metadata.json")
TEST_METADATA_FILE = os.path.join(RESULTS_DIR, "./runs/detect", "test", "test_metadata.json")

# ==========================
# TRAINING PARAMETERS
# ==========================

IMG_SIZE = 640               # Image size for training
EPOCHS = 300                 # Total number of epochs
BATCH_SIZE = 16              # Adjust based on GPU VRAM

# OPTIMAL LEARNING RATES FOR LARGE DATASETS
LR_INITIAL = 0.01          # Lower LR improves stability for 14k+ images
LR_FINAL = 0.01             # Reduced final LR for smoother convergence

PATIENCE = 15                # Early stopping patience

# Device selection
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Pretrained YOLOv26 weights
WEIGHTS = "yolo26s.pt"      
#TRAINED_WEIGHTS = os.path.join("C:\\Users\\Shakiru Sikiru\\ML\\Weapon_Detection\\v26\\runs_1\\detect\\runs\\train\\weapons_yolov26\\train\\weights", "best.pt")  

# ==========================
# REPRODUCIBILITY
# ==========================
RANDOM_SEED = 42

# ==========================
# DETECTION CLASSES
# ==========================
CLASSES = ['Handgun', 'Heavy Weapon', 'Knife', 'Rifle', 'Rocket_Grenade Launcher']


# ==========================
# AUGMENTATION SETTINGS
# (Uncomment to enable)
# ==========================

# augmentations = True
# rect = False
# multi_scale = True
# flipud = 0.5
# fliplr = 0.5
# hsv_h = 0.015
# hsv_s = 0.7
# hsv_v = 0.4
# mosaic = 1.0
# mixup = 0.5