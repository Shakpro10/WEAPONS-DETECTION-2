import os
import cv2
from config import *

# Paths
base_path = DATASET_DIR
splits = ["train", "valid", "test"]

classes = ['Handgun', 'Heavy Weapon', 'Knife', 'Rifle', 'Rocket_Grenade Launcher']

output_base = "./dataset/classifier_dataset"

padding = 0.15  # 15%

def yolo_to_bbox(xc, yc, w, h, img_w, img_h):
    x1 = int((xc - w/2) * img_w)
    y1 = int((yc - h/2) * img_h)
    x2 = int((xc + w/2) * img_w)
    y2 = int((yc + h/2) * img_h)
    return x1, y1, x2, y2

for split in splits:
    img_dir = os.path.join(base_path, split, "images")
    lbl_dir = os.path.join(base_path, split, "labels")

    for cls in classes + ["non_weapon"]:
        os.makedirs(os.path.join(output_base, split, cls), exist_ok=True)

    for img_file in os.listdir(img_dir):
        if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(img_dir, img_file)
        lbl_path = os.path.join(lbl_dir, img_file.replace(".jpg", ".txt"))

        image = cv2.imread(img_path)
        if image is None:
            continue

        h, w, _ = image.shape

        # 🔴 CASE 1: EMPTY LABEL = NEGATIVE IMAGE
        if not os.path.exists(lbl_path) or os.path.getsize(lbl_path) == 0:
            save_path = os.path.join(output_base, split, "non_weapon", img_file)
            cv2.imwrite(save_path, image)
            continue

        # ✅ CASE 2: IMAGE WITH OBJECTS
        with open(lbl_path, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            cls_id, xc, yc, bw, bh = map(float, line.strip().split())
            cls_id = int(cls_id)

            x1, y1, x2, y2 = yolo_to_bbox(xc, yc, bw, bh, w, h)

            # Padding
            pad_w = int((x2 - x1) * padding)
            pad_h = int((y2 - y1) * padding)

            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(w, x2 + pad_w)
            y2 = min(h, y2 + pad_h)

            crop = image[y1:y2, x1:x2]

            save_path = os.path.join(
                output_base, split, classes[cls_id],
                f"{img_file[:-4]}_{i}.jpg"
            )
            cv2.imwrite(save_path, crop)
