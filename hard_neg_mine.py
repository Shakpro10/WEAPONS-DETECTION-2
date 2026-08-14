import os
import cv2
import random
import numpy as np

neg_dirs = [
    "dataset/classifier_dataset/train/non_weapon",
    "dataset/classifier_dataset/valid/non_weapon"
]

all_images = []

for d in neg_dirs:
    for f in os.listdir(d):
        all_images.append(os.path.join(d, f))

total = len(all_images)
augment_count = int(total * 0.8)

output_dir = "dataset/classifier_dataset/train/non_weapon_aug"
os.makedirs(output_dir, exist_ok=True)

def augment(img):
    choice = random.choice([
        "flip", "rotate", "blur", "hsv", "translate"
    ])

    if choice == "flip":
        return cv2.flip(img, 1)

    elif choice == "rotate":
        angle = random.randint(-15, 15)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        return cv2.warpAffine(img, M, (w, h))

    elif choice == "blur":
        return cv2.GaussianBlur(img, (5,5), 0)

    elif choice == "hsv":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = hsv[:,:,1] * random.uniform(0.7, 1.3)
        hsv[:,:,2] = hsv[:,:,2] * random.uniform(0.7, 1.3)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    elif choice == "translate":
        tx, ty = random.randint(-10,10), random.randint(-10,10)
        M = np.float32([[1,0,tx],[0,1,ty]])
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

for i in range(augment_count):
    img_path = random.choice(all_images)
    img = cv2.imread(img_path)

    aug = augment(img)

    save_path = os.path.join(output_dir, f"aug_neg_{i}.jpg")
    cv2.imwrite(save_path, aug)
print(f"✅ Augmented {augment_count} negative samples saved to {output_dir}")