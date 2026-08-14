# =========================
# IMPORTS
# =========================
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score,
    f1_score, fbeta_score, accuracy_score,
)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, random
from collections import defaultdict


# =========================
# CLASS ORDER — MATCHES DETECTOR YAML & train_classifier.py
# =========================
WEAPON_CLASSES = [
    "Handgun",                  # idx 0
    "Heavy Weapon",             # idx 1
    "Knife",                    # idx 2
    "Rifle",                    # idx 3
    "Rocket_Grenade Launcher",  # idx 4
]
REJECT_CLASS = "non_weapon"     # idx 5
ALL_CLASSES  = WEAPON_CLASSES + [REJECT_CLASS]
NUM_CLASSES  = len(ALL_CLASSES)     # 6
REJECT_IDX   = NUM_CLASSES - 1     # 5

# ── CONFIGURATION ────────────────────────────────────────────────────
MODEL_PATH      = "classifier_weights/best_model.pth"
TEST_DIR        = "dataset/classifier_dataset/test"
BATCH_SIZE      = 32
IMAGES_PER_CLS  = 20
OUTPUT_DIR      = "classifier_weights/test_results"
FBETA_BETA      = 0.5
CONF_THRESHOLD  = 0.50      # crops below this confidence are also rejected

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLASS_COLORS = [
    "#E63946",  # Handgun
    "#F4A261",  # Heavy Weapon
    "#2A9D8F",  # Knife
    "#8338EC",  # Rifle
    "#FB5607",  # Rocket_Grenade Launcher
    "#457B9D",  # non_weapon (reject)
]


# =========================
# HELPERS
# =========================
def fix_class_order(dataset):
    """Same function as train_classifier.py — must be kept in sync."""
    found = set(dataset.classes)
    for cls in ALL_CLASSES:
        if cls not in found:
            raise ValueError(
                f"Expected class folder '{cls}' not found.\n"
                f"Found: {sorted(found)}"
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


def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)


# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving plots to: {os.path.abspath(OUTPUT_DIR)}/")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── TRANSFORM ────────────────────────────────────────────────────
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # ── DATASET ──────────────────────────────────────────────────────
    test_data = fix_class_order(
        datasets.ImageFolder(TEST_DIR, transform=test_transform)
    )
    test_loader = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=8, pin_memory=True,
    )

    print("\nClass -> Index mapping (fixed, matches detector YAML):")
    for cls, idx in test_data.class_to_idx.items():
        tag = " <- REJECT class" if idx == REJECT_IDX else ""
        print(f"  [{idx}]  {cls}{tag}")

    # ── MODEL ─────────────────────────────────────────────────────────
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features, NUM_CLASSES
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"\nLoaded model: {MODEL_PATH}")

    # ── INFERENCE ────────────────────────────────────────────────────
    softmax       = nn.Softmax(dim=1)
    all_preds     = []      # raw model prediction (0-5)
    all_preds_gated = []    # after confidence gate (low conf -> REJECT_IDX)
    all_labels    = []
    all_confs     = []

    # For image grids: grouped by TRUE class
    class_samples = defaultdict(list)  # true_idx -> [(img, pred_idx, conf)]

    print("\nRunning inference...")
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs   = imgs.to(device)
            logits = model(imgs)
            probs  = softmax(logits)
            confs, predicted = torch.max(probs, 1)

            for img, true_lbl, pred_lbl, conf_val, prob_row in zip(
                imgs.cpu(), labels, predicted.cpu(), confs.cpu(), probs.cpu()
            ):
                raw_pred  = pred_lbl.item()
                conf      = conf_val.item()
                true_idx  = true_lbl.item()

                # Gated prediction:
                #   - if model predicts REJECT_IDX (5) -> rejected
                #   - if max confidence < threshold   -> rejected (low certainty)
                #   - otherwise pass through as weapon class
                if raw_pred == REJECT_IDX or conf < CONF_THRESHOLD:
                    gated_pred = REJECT_IDX
                else:
                    gated_pred = raw_pred

                all_preds.append(raw_pred)
                all_preds_gated.append(gated_pred)
                all_labels.append(true_idx)
                all_confs.append(conf)

                class_samples[true_idx].append((img, gated_pred, conf))

    all_preds       = np.array(all_preds)
    all_preds_gated = np.array(all_preds_gated)
    all_labels      = np.array(all_labels)
    WEAPON_INDICES  = list(range(len(WEAPON_CLASSES)))  # [0,1,2,3,4]

    # ── METRICS ──────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("OVERALL TEST RESULTS")
    print("="*65)

    # 1. Full accuracy (all 6 classes including reject)
    full_acc = accuracy_score(all_labels, all_preds_gated) * 100
    print(f"\n  Overall accuracy (all 6 classes): {full_acc:.2f}%")

    # 2. Weapon-only metrics (only samples whose TRUE label is a weapon)
    weapon_mask     = all_labels < len(WEAPON_CLASSES)
    w_labels        = all_labels[weapon_mask]
    w_preds_gated   = np.where(
        all_preds_gated[weapon_mask] < len(WEAPON_CLASSES),
        all_preds_gated[weapon_mask], -1
    )

    w_precision = precision_score(
        w_labels, w_preds_gated,
        labels=WEAPON_INDICES, average="macro", zero_division=0,
    )
    w_recall = recall_score(
        w_labels, w_preds_gated,
        labels=WEAPON_INDICES, average="macro", zero_division=0,
    )
    w_f1 = f1_score(
        w_labels, w_preds_gated,
        labels=WEAPON_INDICES, average="macro", zero_division=0,
    )
    w_fbeta = fbeta_score(
        w_labels, w_preds_gated,
        beta=FBETA_BETA, labels=WEAPON_INDICES,
        average="macro", zero_division=0,
    )
    w_acc = accuracy_score(w_labels, np.where(w_preds_gated >= 0, w_preds_gated, -1)) * 100

    print(f"\n  Weapon classification (true weapon crops only, idx 0-4):")
    print(f"    Accuracy          : {w_acc:.2f}%")
    print(f"    Macro Precision   : {w_precision:.4f}")
    print(f"    Macro Recall      : {w_recall:.4f}")
    print(f"    Macro F1          : {w_f1:.4f}")
    print(f"    Macro F{FBETA_BETA} (beta={FBETA_BETA}): {w_fbeta:.4f}")

    # 3. Reject gate: how well does the classifier filter false positives?
    reject_mask      = all_preds_gated == REJECT_IDX
    total_rejected   = np.sum(reject_mask)
    true_rejects     = np.sum(all_labels[reject_mask] == REJECT_IDX)
    actual_nonweapon = np.sum(all_labels == REJECT_IDX)
    gate_precision   = true_rejects / total_rejected  if total_rejected   > 0 else 0.0
    gate_recall      = true_rejects / actual_nonweapon if actual_nonweapon > 0 else 0.0
    # False weapon reject rate: real weapons wrongly rejected
    false_rejects    = np.sum(
        (all_preds_gated == REJECT_IDX) & (all_labels != REJECT_IDX)
    )
    total_weapons    = np.sum(all_labels != REJECT_IDX)
    false_reject_rate = false_rejects / total_weapons if total_weapons > 0 else 0.0

    print(f"\n  Reject gate (false-positive filter) @ conf >= {CONF_THRESHOLD}:")
    print(f"    Total crops rejected      : {total_rejected}")
    print(f"    Correctly rejected (TP)   : {true_rejects}")
    print(f"    Gate precision            : {gate_precision:.4f}  "
          f"(of rejected crops, how many were actually non_weapon)")
    print(f"    Gate recall               : {gate_recall:.4f}  "
          f"(of all non_weapon, how many got rejected)")
    print(f"    False weapon reject rate  : {false_reject_rate:.4f}  "
          f"({false_rejects} real weapons wrongly rejected)")

    # 4. Full per-class report
    print(f"\n  Full per-class report:")
    print(classification_report(
        all_labels, all_preds_gated,
        target_names=ALL_CLASSES, zero_division=0,
    ))

    # ── PLOTS ────────────────────────────────────────────────────────
    _plot_metrics_bar(all_labels, all_preds_gated)
    _plot_confusion_matrix(all_labels, all_preds_gated)
    _plot_image_grids(class_samples)

    print(f"\nAll plots saved to '{OUTPUT_DIR}/'")
    print("Done!")


# =========================
# PLOT 1 — METRICS BAR CHART
# =========================
def _plot_metrics_bar(all_labels, all_preds_gated):
    precision_vals = precision_score(
        all_labels, all_preds_gated,
        labels=list(range(NUM_CLASSES)),
        average=None, zero_division=0,
    )
    recall_vals = recall_score(
        all_labels, all_preds_gated,
        labels=list(range(NUM_CLASSES)),
        average=None, zero_division=0,
    )
    f1_vals = f1_score(
        all_labels, all_preds_gated,
        labels=list(range(NUM_CLASSES)),
        average=None, zero_division=0,
    )

    short_names = ["Handgun", "Heavy\nWeapon", "Knife",
                   "Rifle", "Rocket\nGrenade", "non_weapon\n(reject)"]
    metrics     = {"Precision": precision_vals,
                   "Recall": recall_vals, "F1": f1_vals}
    n_cls       = len(ALL_CLASSES)
    x           = np.arange(n_cls)
    width       = 0.25
    bar_colors  = ["#264653", "#2A9D8F", "#E9C46A"]

    fig, ax = plt.subplots(figsize=(15, 6))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#16213E")

    for i, (name, vals) in enumerate(metrics.items()):
        bars = ax.bar(x + i * width, vals, width, label=name,
                      color=bar_colors[i], edgecolor="white",
                      linewidth=0.4, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=7, color="white")

    # Shade reject column background
    ax.axvspan(n_cls - 1 - 0.4, n_cls - 0.6 + width * 2,
               color="#FFFFFF", alpha=0.04, zorder=0)
    ax.axvline(n_cls - 1 - 0.45, color="#555", linewidth=1,
               linestyle="--", zorder=1)

    ax.set_xticks(x + width)
    ax.set_xticklabels(short_names, rotation=15, ha="right",
                       fontsize=9, color="white")
    ax.set_ylabel("Score", color="white", fontsize=11)
    ax.set_title("Per-Class Precision / Recall / F1\n"
                 "(right of dashed line = reject/non_weapon gate)",
                 color="white", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, 1.18)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.yaxis.grid(True, color="#333", zorder=0)
    ax.legend(fontsize=10, facecolor="#1A1A2E",
              labelcolor="white", edgecolor="#444")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "metrics_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [Saved] {path}")


# =========================
# PLOT 2 — CONFUSION MATRIX
# =========================
def _plot_confusion_matrix(all_labels, all_preds_gated):
    cm      = confusion_matrix(all_labels, all_preds_gated,
                               labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    short_names = ["Handgun", "Heavy\nWeapon", "Knife",
                   "Rifle", "Rocket\nGrenade", "non_weapon\n(reject)"]

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    fig.patch.set_facecolor("#1A1A2E")

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Confusion Matrix — Raw Counts",
         "Confusion Matrix — Row Normalised"],
        [".0f", ".2f"],
    ):
        ax.set_facecolor("#16213E")
        im  = ax.imshow(data, interpolation="nearest",
                        cmap="YlOrRd", vmin=0, vmax=data.max())
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        thresh = data.max() / 2
        for i in range(len(short_names)):
            for j in range(len(short_names)):
                ax.text(j, i, format(data[i, j], fmt),
                        ha="center", va="center", fontsize=8,
                        color="white" if data[i, j] < thresh else "black")

        ax.set_xticks(range(len(short_names)))
        ax.set_yticks(range(len(short_names)))
        ax.set_xticklabels(short_names, color="white", fontsize=8, rotation=15)
        ax.set_yticklabels(short_names, color="white", fontsize=8)
        ax.set_xlabel("Predicted", color="white", fontsize=11)
        ax.set_ylabel("True", color="white", fontsize=11)
        ax.set_title(title, color="white", fontsize=12,
                     fontweight="bold", pad=10)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#444")

        # Highlight the reject column/row
        for spine_type in ["left", "right", "top", "bottom"]:
            ax.spines[spine_type].set_color("#444")
        ax.axvline(NUM_CLASSES - 1.5, color="#FFD700",
                   linewidth=1.5, linestyle="--")
        ax.axhline(NUM_CLASSES - 1.5, color="#FFD700",
                   linewidth=1.5, linestyle="--")

    plt.suptitle("Confusion Matrices\n"
                 "(gold lines separate reject/non_weapon gate from weapon classes)",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [Saved] {path}")


# =========================
# PLOT 3 — IMAGE GRIDS
# =========================
def _plot_image_grids(class_samples):
    """
    One figure per TRUE class.  20 images (4 rows x 5 cols).
    Green border  = correct prediction
    Red border    = wrong prediction
    Gold border   = correctly REJECTED (true non_weapon caught by gate)
    Caption: True: X | Pred: Y | Conf: Z%
    """
    COLS = 5
    ROWS = IMAGES_PER_CLS // COLS

    for true_idx, samples in class_samples.items():
        true_name  = ALL_CLASSES[true_idx]
        cls_colour = CLASS_COLORS[true_idx]

        random.shuffle(samples)
        chosen = samples[:IMAGES_PER_CLS]
        while len(chosen) < IMAGES_PER_CLS:
            chosen.append(random.choice(samples))

        fig, axes = plt.subplots(ROWS, COLS,
                                 figsize=(COLS * 3.2, ROWS * 3.6))
        fig.patch.set_facecolor("#0D0D0D")

        fig.suptitle(f"True Class: {true_name}  |  {IMAGES_PER_CLS} samples",
                     fontsize=15, fontweight="bold",
                     color=cls_colour, y=1.01)

        # Legend
        patches = [
            mpatches.Patch(color="#00C896", label="Correct weapon class"),
            mpatches.Patch(color="#FF4444", label="Wrong prediction"),
            mpatches.Patch(color="#FFD700",
                           label="Correctly rejected (non_weapon gate)"),
            mpatches.Patch(color="#FF8C00",
                           label="Wrongly rejected (weapon lost to gate)"),
        ]
        fig.legend(handles=patches, loc="upper right", fontsize=8,
                   facecolor="#1A1A1A", labelcolor="white",
                   edgecolor="#555", framealpha=0.9)

        for i, ax in enumerate(axes.flat):
            img_tensor, pred_idx, conf = chosen[i]
            pred_name = ALL_CLASSES[pred_idx]
            is_reject_class = (true_idx == REJECT_IDX)

            # Border colour logic
            if is_reject_class and pred_idx == REJECT_IDX:
                border_col = "#FFD700"   # true non_weapon, correctly rejected
            elif not is_reject_class and pred_idx == true_idx:
                border_col = "#00C896"   # weapon, correctly classified
            elif not is_reject_class and pred_idx == REJECT_IDX:
                border_col = "#FF8C00"   # weapon wrongly rejected by gate
            else:
                border_col = "#FF4444"   # wrong weapon class or non_weapon passed

            img_np = denormalize(img_tensor).permute(1, 2, 0).numpy()
            ax.imshow(img_np)
            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_edgecolor(border_col)
                spine.set_linewidth(3)

            # "REJECTED" label if gated out
            if pred_idx == REJECT_IDX and not is_reject_class:
                ax.set_title("REJECTED", fontsize=8,
                             color="#FF8C00", pad=2)
            elif pred_idx == REJECT_IDX and is_reject_class:
                ax.set_title("REJECTED ✓", fontsize=8,
                             color="#FFD700", pad=2)

            caption = (
                f"True: {true_name}\n"
                f"Pred: {pred_name}\n"
                f"Conf: {conf*100:.1f}%"
            )
            ax.set_xlabel(caption, fontsize=7,
                          color="white", labelpad=3)
            ax.set_facecolor("#0D0D0D")

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        safe_name = true_name.replace(" ", "_").replace("/", "_")
        path = os.path.join(OUTPUT_DIR, f"samples_{safe_name}.png")
        plt.savefig(path, dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"  [Saved] {path}")


if __name__ == "__main__":
    main()