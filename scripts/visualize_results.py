"""Visualization utilities for Brain Tumor Segmentation pipeline.
1. Plot loss and validation Dice curves from training logs.
2. Generate qualitative PNG overlays for Best, Median, Worst, and Stage-1 failure cases.
"""
import argparse
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import csv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data import to_regions
from infer import load_models, segment


def plot_curves(history_csv, out_png="results/curves.png"):
    """Plot training loss and validation Dice curves from history.csv."""
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    if not os.path.exists(history_csv):
        print(f"Warning: {history_csv} does not exist. Skipping curve plot.")
        return

    epochs, losses = [], []
    val_wt, val_tc, val_et = [], [], []

    with open(history_csv, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            epochs.append(int(r["epoch"]))
            losses.append(float(r["train_loss"]))
            if "val_WT" in r:
                val_wt.append(float(r["val_WT"]))
            if "val_TC" in r:
                val_tc.append(float(r["val_TC"]))
            if "val_ET" in r:
                val_et.append(float(r["val_ET"]))

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = "tab:red"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color=color)
    ax1.plot(epochs, losses, color=color, label="Train Loss", lw=2)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Validation Dice", color="tab:blue")
    if val_wt:
        ax2.plot(epochs, val_wt, label="Val WT", lw=2, color="tab:green")
    if val_tc:
        ax2.plot(epochs, val_tc, label="Val TC", lw=2, color="tab:orange")
    if val_et:
        ax2.plot(epochs, val_et, label="Val ET", lw=2, color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_ylim(0, 1.05)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title(f"Training History ({os.path.basename(os.path.dirname(history_csv))})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Curve plot saved to {out_png}")


def overlay_mask_on_slice(img_slice, mask_wt, mask_tc, mask_et, alpha=0.4):
    """Create RGB overlay on grayscale MRI slice:
    WT: Green, TC: Yellow (Red+Green), ET: Red
    """
    # Normalize img to 0..1
    lo, hi = img_slice.min(), img_slice.max()
    norm_img = (img_slice - lo) / (hi - lo + 1e-8)
    rgb = np.stack([norm_img, norm_img, norm_img], axis=-1)

    # Colors
    # Edema (WT \ TC): Green
    edema = mask_wt & (~mask_tc)
    # Necrotic (TC \ ET): Yellow
    necrotic = mask_tc & (~mask_et)
    # Enhancing (ET): Red
    et = mask_et

    # Apply colors
    # Green for edema
    rgb[edema, 0] = rgb[edema, 0] * (1 - alpha)
    rgb[edema, 1] = rgb[edema, 1] * (1 - alpha) + alpha * 1.0
    rgb[edema, 2] = rgb[edema, 2] * (1 - alpha)

    # Yellow for necrotic (R=1, G=1, B=0)
    rgb[necrotic, 0] = rgb[necrotic, 0] * (1 - alpha) + alpha * 1.0
    rgb[necrotic, 1] = rgb[necrotic, 1] * (1 - alpha) + alpha * 0.9
    rgb[necrotic, 2] = rgb[necrotic, 2] * (1 - alpha)

    # Red for ET
    rgb[et, 0] = rgb[et, 0] * (1 - alpha) + alpha * 1.0
    rgb[et, 1] = rgb[et, 1] * (1 - alpha)
    rgb[et, 2] = rgb[et, 2] * (1 - alpha)

    return np.clip(rgb, 0, 1)


def save_qualitative_overlays(results_csv, cache_dir, s1_path, s2_path, out_dir="results/overlays", et_label=4):
    os.makedirs(out_dir, exist_ok=True)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    coarse, fine = load_models(s1_path, s2_path, dev)

    with open(results_csv, "r") as f:
        reader = list(csv.DictReader(f))

    # Sort cases by WT Dice
    valid = [r for r in reader if float(r.get("WT_Dice", 0)) > 0]
    valid_sorted = sorted(valid, key=lambda x: float(x["WT_Dice"]))

    selected = {}
    if valid_sorted:
        selected["best"] = valid_sorted[-1]
        selected["median"] = valid_sorted[len(valid_sorted) // 2]
        selected["worst"] = valid_sorted[0]

    # Stage 1 failure: found nothing or lowest recall
    s1_failures = [r for r in reader if r.get("S1_FoundNothing", "").lower() == "true"]
    if s1_failures:
        selected["s1_failure"] = s1_failures[0]
    else:
        # Lowest S1 recall
        selected["s1_failure"] = min(reader, key=lambda x: float(x.get("S1_Recall", 1.0)))

    print(f"Generating qualitative overlays for: {list(selected.keys())}")

    for tag, row in selected.items():
        cid = row["case"]
        img = np.load(os.path.join(cache_dir, f"{cid}_img.npy")).astype(np.float32)
        lab = np.load(os.path.join(cache_dir, f"{cid}_lab.npy"))

        pred, bb, wt_coarse = segment(coarse, fine, img, dev, return_stage1=True)
        gt = to_regions(lab, et_label=et_label) > 0.5

        # Find best axial slice showing tumor (max WT voxels)
        wt_sum = gt[0].sum(axis=(0, 1))
        if wt_sum.max() > 0:
            slice_idx = int(np.argmax(wt_sum))
        else:
            slice_idx = img.shape[-1] // 2

        flair_slice = img[0, :, :, slice_idx]
        t1ce_slice = img[2, :, :, slice_idx]

        gt_overlay_flair = overlay_mask_on_slice(flair_slice, gt[0, :, :, slice_idx], gt[1, :, :, slice_idx], gt[2, :, :, slice_idx])
        pred_overlay_flair = overlay_mask_on_slice(flair_slice, pred[0, :, :, slice_idx], pred[1, :, :, slice_idx], pred[2, :, :, slice_idx])
        pred_overlay_t1ce = overlay_mask_on_slice(t1ce_slice, pred[0, :, :, slice_idx], pred[1, :, :, slice_idx], pred[2, :, :, slice_idx])

        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        axes[0].imshow(flair_slice, cmap="gray")
        axes[0].set_title(f"FLAIR Slice {slice_idx}")
        axes[0].axis("off")

        axes[1].imshow(gt_overlay_flair)
        axes[1].set_title(f"Ground Truth (Green=WT, Yellow=TC, Red=ET)")
        axes[1].axis("off")

        axes[2].imshow(pred_overlay_flair)
        axes[2].set_title(f"Prediction (FLAIR) | WT Dice: {float(row.get('WT_Dice', 0)):.3f}")
        axes[2].axis("off")

        axes[3].imshow(pred_overlay_t1ce)
        axes[3].set_title(f"Prediction (T1ce) | ET Dice: {float(row.get('ET_Dice', 0)):.3f}")
        axes[3].axis("off")

        out_path = os.path.join(out_dir, f"{tag}_{cid}_slice{slice_idx}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="path to results.csv")
    ap.add_argument("--cache", default=None, help="path to preprocessed cache")
    ap.add_argument("--s1", default=None, help="path to s1 best.pt")
    ap.add_argument("--s2", default=None, help="path to s2 best.pt")
    ap.add_argument("--history", default=None, help="path to history.csv for curve plotting")
    ap.add_argument("--out_dir", default="results/overlays")
    ap.add_argument("--curve_out", default="results/curves.png")
    args = ap.parse_args()

    if args.history:
        plot_curves(args.history, args.curve_out)

    if args.csv and args.cache and args.s1 and args.s2:
        save_qualitative_overlays(args.csv, args.cache, args.s1, args.s2, args.out_dir)
