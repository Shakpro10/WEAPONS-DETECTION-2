# =========================
# IMPORTS
# =========================
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_score, recall_score,
    fbeta_score, classification_report,
)
import numpy as np
import os


# =========================
# CLASS ORDER — MUST MATCH DETECTOR YAML
# =========================
# Detector YAML: ['Handgun', 'Heavy Weapon', 'Knife', 'Rifle', 'Rocket_Grenade Launcher']
#
# We fix indices 0-4 to mirror the detector exactly.
# non_weapon is forced to index 5 (last) so it never collides with detector indices.
#
# WHY THIS MATTERS:
#   ImageFolder default (alphabetical) gives: Handgun=0, Heavy Weapon=1, Knife=2,
#   non_weapon=3, Rifle=4, Rocket_Grenade=5.
#   That puts non_weapon at index 3, silently shifting Rifle -> 4 and
#   Rocket_Grenade -> 5.  Every Rifle prediction would be misread as
#   non_weapon (index 3) by the detector pipeline — a silent but fatal bug.
#
# At inference:
#   prediction  0-4  ->  weapon confirmed, class index matches detector
#   prediction  5    ->  REJECTED (crop is a false positive, discarded)
#   confidence < CONF_THRESHOLD -> also REJECTED regardless of class

WEAPON_CLASSES = [
    "Handgun",                  # idx 0  <- matches detector idx 0
    "Heavy Weapon",             # idx 1  <- matches detector idx 1
    "Knife",                    # idx 2  <- matches detector idx 2
    "Rifle",                    # idx 3  <- matches detector idx 3
    "Rocket_Grenade Launcher",  # idx 4  <- matches detector idx 4
]
REJECT_CLASS = "non_weapon"     # idx 5  <- reject / false-positive filter
ALL_CLASSES  = WEAPON_CLASSES + [REJECT_CLASS]
NUM_CLASSES  = len(ALL_CLASSES)     # 6
REJECT_IDX   = NUM_CLASSES - 1     # 5


# =========================
# HELPER — FORCE CLASS ORDER
# =========================
def fix_class_order(dataset):
    """
    Override ImageFolder's alphabetical class_to_idx so indices 0-4
    match the detector YAML and non_weapon lands at index 5.
    Call this immediately after creating every ImageFolder.
    """
    found = set(dataset.classes)
    for cls in ALL_CLASSES:
        if cls not in found:
            raise ValueError(
                f"Expected class folder '{cls}' not found in dataset.\n"
                f"Found: {sorted(found)}\n"
                f"Check that folder names match ALL_CLASSES exactly "
                f"(case-sensitive, including spaces)."
            )

    new_class_to_idx = {cls: idx for idx, cls in enumerate(ALL_CLASSES)}

    new_samples = []
    for path, _ in dataset.imgs:
        folder_name = os.path.basename(os.path.dirname(path))
        if folder_name in new_class_to_idx:
            new_samples.append((path, new_class_to_idx[folder_name]))

    dataset.class_to_idx = new_class_to_idx
    dataset.classes      = ALL_CLASSES
    dataset.samples      = new_samples
    dataset.imgs         = new_samples
    dataset.targets      = [s[1] for s in new_samples]
    return dataset


# =========================
# MAIN
# =========================
def main():

    # ── CONFIGURATION ────────────────────────────────────────────────
    BATCH_SIZE          = 32
    EPOCHS              = 50
    MODEL_SAVE_PATH     = "classifier_weights/best_model.pth"
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    FREEZE_EPOCHS       = 5       # Train head only first, then unfreeze backbone
    EARLY_STOP_PATIENCE = 7
    LR                  = 1e-4
    FBETA_BETA          = 0.5     # precision weighted 2x over recall

    # ── DEVICE ───────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── TRANSFORMS ───────────────────────────────────────────────────
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # ── DATASETS ─────────────────────────────────────────────────────
    train_data = fix_class_order(
        datasets.ImageFolder("dataset/classifier_dataset/train",
                             transform=train_transform)
    )
    val_data = fix_class_order(
        datasets.ImageFolder("dataset/classifier_dataset/valid",
                             transform=val_transform)
    )

    print("\nClass -> Index mapping (FIXED to match detector YAML):")
    for cls, idx in train_data.class_to_idx.items():
        tag = " <- REJECT (false-positive gate)" if idx == REJECT_IDX \
              else " <- matches detector"
        print(f"  [{idx}]  {cls:<30}{tag}")

    # ── CLASS WEIGHTS (imbalance fix) ─────────────────────────────────
    # Train counts:
    #   Handgun              -> 4,642
    #   Heavy Weapon         -> 2,012   <- minority, boost needed
    #   Knife                -> 7,362
    #   Rifle                -> 9,226
    #   Rocket_Grenade Lnchr -> 1,600   <- smallest weapon class
    #   non_weapon           -> 10,978  <- majority / reject class
    #
    # Formula: weight_i = N / (C * n_i)
    targets       = torch.tensor(train_data.targets)
    class_counts  = torch.bincount(targets, minlength=NUM_CLASSES)
    total_samples = len(targets)
    class_weights = total_samples / (NUM_CLASSES * class_counts.float())
    class_weights = class_weights.to(device)

    print("\nComputed class weights (higher = more loss attention):")
    for cls, idx in train_data.class_to_idx.items():
        print(f"  {cls:<30} count={class_counts[idx]:>6}  "
              f"weight={class_weights[idx]:.4f}")

    # ── DATALOADERS ───────────────────────────────────────────────────
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=8, pin_memory=True, persistent_workers=True,
    )

    # ── MODEL ─────────────────────────────────────────────────────────
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features, NUM_CLASSES
    )
    model = model.to(device)

    # ── PHASE 1: FREEZE BACKBONE ──────────────────────────────────────
    for param in model.features.parameters():
        param.requires_grad = False
    print(f"\nBackbone frozen for first {FREEZE_EPOCHS} epochs.")

    # ── LOSS / OPTIMIZER / SCHEDULER ─────────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5,
    )

    # ── TRACKING ─────────────────────────────────────────────────────
    best_fbeta        = 0.0
    epochs_no_improve = 0
    WEAPON_INDICES    = list(range(len(WEAPON_CLASSES)))  # [0,1,2,3,4]

    print("\n" + "="*62)
    print("TRAINING START")
    print(f"  Weapon output classes (idx 0-4): {WEAPON_CLASSES}")
    print(f"  Reject class          (idx   5): {REJECT_CLASS}")
    print(f"  Non-weapon crops used as hard negatives during training.")
    print(f"  At inference: idx==5 or low confidence -> crop rejected.")
    print("="*62)

    # ── TRAINING LOOP ─────────────────────────────────────────────────
    for epoch in range(EPOCHS):

        # Phase 2: unfreeze backbone
        if epoch == FREEZE_EPOCHS:
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LR / 10, weight_decay=1e-4,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=3, factor=0.5,
            )
            print(f"\nBackbone unfrozen at epoch {epoch+1}. LR={LR/10:.1e}")

        # ---- TRAIN --------------------------------------------------
        model.train()
        train_loss    = 0
        train_correct = 0
        train_total   = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds  = model(imgs)
            loss   = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss    += loss.item()
            _, predicted   = torch.max(preds, 1)
            train_total   += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc      = 100 * train_correct / train_total

        # ---- VALIDATE -----------------------------------------------
        model.eval()
        val_loss   = 0
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs)
                val_loss += criterion(preds, labels).item()
                _, predicted = torch.max(preds, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  LR: {current_lr:.2e}")

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)
        val_acc    = 100 * np.mean(all_preds == all_labels)

        # ---- WEAPON-ONLY METRICS (indices 0-4) ----------------------
        # Filter to samples whose TRUE label is a weapon class.
        # non_weapon (idx 5) trains the reject gate but is excluded
        # from weapon classification metrics.
        weapon_mask     = all_labels < len(WEAPON_CLASSES)
        w_labels        = all_labels[weapon_mask]
        # Any prediction of 5 (reject) on a weapon sample counts as wrong
        w_preds_clamped = np.where(
            all_preds[weapon_mask] < len(WEAPON_CLASSES),
            all_preds[weapon_mask], -1
        )

        precision = precision_score(
            w_labels, w_preds_clamped,
            labels=WEAPON_INDICES, average="macro", zero_division=0,
        )
        recall = recall_score(
            w_labels, w_preds_clamped,
            labels=WEAPON_INDICES, average="macro", zero_division=0,
        )
        fbeta = fbeta_score(
            w_labels, w_preds_clamped,
            beta=FBETA_BETA, labels=WEAPON_INDICES,
            average="macro", zero_division=0,
        )

        # Reject gate stats: how well does idx-5 filter non_weapon crops?
        reject_mask      = all_preds == REJECT_IDX
        true_rejects     = np.sum(all_labels[reject_mask] == REJECT_IDX)
        total_rejected   = np.sum(reject_mask)
        actual_nonweapon = np.sum(all_labels == REJECT_IDX)
        reject_precision = true_rejects / total_rejected  if total_rejected   > 0 else 0.0
        reject_recall    = true_rejects / actual_nonweapon if actual_nonweapon > 0 else 0.0

        # ---- LOG ----------------------------------------------------
        print(f"\n{'─'*62}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train  ->  Loss: {avg_train_loss:.4f}  Acc: {train_acc:.2f}%")
        print(f"  Val    ->  Loss: {avg_val_loss:.4f}  Acc: {val_acc:.2f}%")
        print(f"  Weapon metrics (idx 0-4, detector-aligned):")
        print(f"    Precision : {precision:.4f}")
        print(f"    Recall    : {recall:.4f}")
        print(f"    F{FBETA_BETA}       : {fbeta:.4f}")
        print(f"  Reject gate performance:")
        print(f"    Rejected {total_rejected:>5} crops  |  "
              f"Gate precision: {reject_precision:.4f}  |  "
              f"Gate recall: {reject_recall:.4f}")

        if (epoch + 1) % 5 == 0:
            print(f"\n  Full per-class report (epoch {epoch+1}):")
            report = classification_report(
                all_labels, all_preds,
                target_names=ALL_CLASSES, zero_division=0,
            )
            for line in report.split("\n"):
                print(f"    {line}")

        # ---- SAVE BEST (weapon F-beta, not reject class) ------------
        if fbeta > best_fbeta:
            best_fbeta        = fbeta
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  Best model saved!  (weapon F{FBETA_BETA}={best_fbeta:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement: {epochs_no_improve}/{EARLY_STOP_PATIENCE}")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping after {epoch+1} epochs.")
            break

    print(f"\nTraining complete!  Best weapon F{FBETA_BETA}: {best_fbeta:.4f}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()