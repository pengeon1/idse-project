"""Full coarse-to-fine inference + evaluation.

  python infer.py --cache cache --s1 runs/s1/best.pt --s2 runs/s2/best.pt --split test
  python infer.py ... --drop_mods t1,t2        # test robustness to missing modalities
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from data import MODS, bbox_of, to_regions
from models import CoarseNet, FineNet, pad_to_multiple, unpad


def compute_hd95(pred, gt, voxel_spacing=(1.0, 1.0, 1.0), empty_penalty=373.13):
    """Compute 95th percentile Hausdorff Distance (in mm/voxels).
    Handles empty predictions and targets explicitly.
    If both empty: 0.0 (perfect agreement on negative class).
    If one empty: empty_penalty (standard maximum volume diameter penalty).
    """
    from scipy import ndimage
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float(empty_penalty)
    s_pred = pred ^ ndimage.binary_erosion(pred)
    s_gt = gt ^ ndimage.binary_erosion(gt)
    # Ensure at least 1 surface voxel
    if s_pred.sum() == 0:
        s_pred = pred
    if s_gt.sum() == 0:
        s_gt = gt
    dt_gt = ndimage.distance_transform_edt(~gt, sampling=voxel_spacing)
    dt_pred = ndimage.distance_transform_edt(~pred, sampling=voxel_spacing)
    dists = np.concatenate([dt_gt[s_pred], dt_pred[s_gt]])
    return float(np.percentile(dists, 95))


def load_models(s1_path, s2_path, dev):
    c1 = torch.load(s1_path, map_location=dev, weights_only=False)
    c2 = torch.load(s2_path, map_location=dev, weights_only=False)
    a1, a2 = c1["args"], c2["args"]
    coarse = CoarseNet(base=a1.get("base") or 8).to(dev)
    dilated_stages = () if a2.get("no_dilated", False) else a2.get("dilated_stages", (3, 4))
    fine = FineNet(base=a2.get("base") or 16, skip_attn=a2.get("skip_attn", "cbam"),
                   dilated_stages=dilated_stages, tf_layers=a2.get("tf_layers", 2)).to(dev)
    coarse.load_state_dict(c1["model"])
    fine.load_state_dict(c2["model"])
    return coarse.eval(), fine.eval()


def largest_component(mask):
    try:
        from scipy import ndimage
    except ImportError:
        return mask
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def apply_scanner_perturbation(img, shift_type):
    """Simulate scanner domain shifts / acquisition differences:
    - gamma_low: simulated hyper-intense scan (gamma = 0.75)
    - gamma_high: simulated hypo-intense scan (gamma = 1.35)
    - bias_field: spatial intensity non-uniformity (B0 inhomogeneity / RF coil gradient)
    - noise: low SNR scanner Gaussian noise
    """
    if not shift_type or shift_type == "none":
        return img
    img = img.copy()
    for m in range(img.shape[0]):
        nz = img[m] != 0
        if not nz.any():
            continue
        v = img[m, nz]
        vmin, vmax = v.min(), v.max()
        norm = (v - vmin) / (vmax - vmin + 1e-8)
        
        if shift_type == "gamma_low":
            norm = np.power(norm, 0.75)
        elif shift_type == "gamma_high":
            norm = np.power(norm, 1.35)
        elif shift_type == "bias_field":
            H, W, D = img[m].shape
            x_coord = np.linspace(-1, 1, H)[:, None, None]
            y_coord = np.linspace(-1, 1, W)[None, :, None]
            bias = 1.0 + 0.25 * (x_coord**2 + y_coord)
            img[m] = np.where(nz, img[m] * bias[nz], 0.0)
            continue
        elif shift_type == "noise":
            noise = np.random.normal(0, 0.15, size=v.shape)
            img[m, nz] = v + noise
            continue
            
        rescaled = norm * (vmax - vmin) + vmin
        rescaled = (rescaled - rescaled.mean()) / (rescaled.std() + 1e-8)
        img[m, nz] = rescaled
    return img


def forward_fine_crop(fine, crop, dev, use_amp=True, tta=False):
    """Run fine model forward pass with optional test-time augmentation (axial/coronal/sagittal flips)."""
    flips = [(), (2,), (3,), (4,)] if tta else [()]
    probs = []
    for f_axes in flips:
        x_flip = torch.flip(crop, dims=f_axes) if f_axes else crop
        x_p, csz = pad_to_multiple(x_flip, 16)
        with torch.autocast(dev, dtype=torch.float16, enabled=use_amp):
            p = unpad(torch.sigmoid(fine(x_p)), csz)
        if f_axes:
            p = torch.flip(p, dims=f_axes)
        probs.append(p)
    return torch.stack(probs).mean(dim=0)[0].float().cpu().numpy()


@torch.no_grad()
def segment(coarse, fine, img, dev, down=2, margin=12, thr=0.5, box_thr=0.35,
            adaptive_margin=False, tta=False, return_stage1=False, single_stage=False):
    """img: float array (4,H,W,D), already z-scored. Returns bool (3,H,W,D): WT, TC, ET."""
    x = torch.from_numpy(np.asarray(img, dtype=np.float32))[None].to(dev)
    shape = x.shape[2:]
    use_amp = dev == "cuda"

    if single_stage:
        # Baseline A7: run FineNet directly on the full padded volume without Stage 1 coarse localization
        prob = forward_fine_crop(fine, x, dev, use_amp=use_amp, tta=tta) > thr
        wt, tc, et = prob
        tc &= wt
        et &= tc
        out = np.stack([wt, tc, et])
        return (out, (np.zeros(3, dtype=int), np.array(shape, dtype=int)), None) if return_stage1 else out

    # ---- stage 1: coarse whole-tumour localisation at low resolution
    xl, sz = pad_to_multiple(F.avg_pool3d(x, down), 8)
    with torch.autocast(dev, dtype=torch.float16, enabled=use_amp):
        p = unpad(torch.sigmoid(coarse(xl)), sz)
    p = F.interpolate(p.float(), size=shape, mode="trilinear", align_corners=False)
    p_np = p[0, 0].cpu().numpy()
    wt_coarse = largest_component(p_np > thr)

    if adaptive_margin:
        # Detect boundary uncertainty: check volume with p in [box_thr, thr]
        # Infiltrative edema produces soft boundary voxels between 0.30 and 0.50
        box_candidate = largest_component(p_np > box_thr)
        diff_voxels = int((box_candidate & (~wt_coarse)).sum())
        ratio = diff_voxels / (wt_coarse.sum() + 1e-5)
        # Adaptively widen margin from 12 up to 24 voxels if boundary is diffuse
        dyn_margin = min(24, int(margin + 12 * min(1.0, ratio * 2)))
        bb = bbox_of(box_candidate, margin=dyn_margin, min_size=64)
    else:
        bb = bbox_of(wt_coarse, margin=margin, min_size=64)

    out = np.zeros((3,) + tuple(shape), dtype=bool)
    if bb is None:                                        # nothing found
        return (out, None, wt_coarse) if return_stage1 else out
    lo, hi = bb

    # ---- stage 2: fine segmentation of the crop (with optional TTA)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    crop = x[(slice(None), slice(None)) + sl]
    prob = forward_fine_crop(fine, crop, dev, use_amp=use_amp, tta=tta) > thr
    wt, tc, et = prob
    tc &= wt                                              # enforce ET <= TC <= WT
    et &= tc
    out[(slice(None),) + sl] = np.stack([wt, tc, et])
    return (out, (lo, hi), wt_coarse) if return_stage1 else out


def dice(a, b, eps=1e-5):
    a, b = a.astype(bool), b.astype(bool)
    if a.sum() + b.sum() == 0:
        return 1.0
    return float(2 * (a & b).sum() / (a.sum() + b.sum() + eps))


def main():
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--s1", default=None, help="path to stage 1 weights (optional if --single_stage)")
    ap.add_argument("--s2", required=True, help="path to stage 2 weights")
    ap.add_argument("--split_file", default=None, help="split.json from training (default: next to --s2)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--et_label", type=int, default=4)
    ap.add_argument("--margin", type=int, default=12)
    ap.add_argument("--drop_mods", default="", help="comma list of modalities to zero out, e.g. t1,t2")
    ap.add_argument("--single_stage", action="store_true", help="A7: single stage FineNet without stage 1")
    ap.add_argument("--adaptive_margin", action="store_true", help="Uncertainty-aware adaptive margin for Stage 1")
    ap.add_argument("--box_thr", type=float, default=0.35, help="Coarse threshold for bounding box candidate extraction")
    ap.add_argument("--tta", action="store_true", help="Enable 4-fold test-time augmentation")
    ap.add_argument("--shift_type", default="none", choices=["none", "gamma_low", "gamma_high", "bias_field", "noise"],
                    help="Simulate scanner domain shift / intensity non-uniformity")
    ap.add_argument("--all_cases", action="store_true", help="Evaluate on all cases found in cache (useful for cross-dataset transfer)")
    ap.add_argument("--out_csv", default="results.csv")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.s1:
        coarse, fine = load_models(a.s1, a.s2, dev)
    else:
        c2 = torch.load(a.s2, map_location=dev, weights_only=False)
        a2 = c2["args"]
        dilated_stages = () if a2.get("no_dilated", False) else a2.get("dilated_stages", (3, 4))
        fine = FineNet(base=a2.get("base") or 16, skip_attn=a2.get("skip_attn", "cbam"),
                       dilated_stages=dilated_stages, tf_layers=a2.get("tf_layers", 2)).to(dev)
        fine.load_state_dict(c2["model"])
        coarse = None

    if a.all_cases:
        cases = list_cases(a.cache)
    else:
        split_file = a.split_file or os.path.join(os.path.dirname(a.s2), "split.json")
        if os.path.exists(split_file):
            split_dict = json.load(open(split_file))
            if a.split in split_dict:
                cases = [c for c in split_dict[a.split] if os.path.exists(os.path.join(a.cache, f"{c}_img.npy"))]
                if not cases:
                    cases = list_cases(a.cache)
            else:
                cases = list_cases(a.cache)
        else:
            cases = list_cases(a.cache)

    drop = [MODS.index(m) for m in a.drop_mods.split(",") if m]

    rows = []
    s1_recalls = []
    s1_contains = []
    s1_empty_count = 0

    for c in cases:
        img = np.load(os.path.join(a.cache, f"{c}_img.npy")).astype(np.float32)
        lab = np.load(os.path.join(a.cache, f"{c}_lab.npy"))
        for i in drop:
            img[i] = 0

        # Optional scanner perturbation stress-test
        if a.shift_type != "none":
            img = apply_scanner_perturbation(img, a.shift_type)

        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        pred, bb, wt_coarse = segment(
            coarse, fine, img, dev,
            margin=a.margin,
            box_thr=a.box_thr,
            adaptive_margin=a.adaptive_margin,
            tta=a.tta,
            return_stage1=True,
            single_stage=a.single_stage
        )
        t_elapsed = time.time() - t0
        peak_mb = torch.cuda.max_memory_allocated() / 1e6 if dev == "cuda" else 0.0

        gt = to_regions(lab, a.et_label) > 0.5
        gt_wt = gt[0]

        # Stage 1 diagnostics
        if a.single_stage or wt_coarse is None:
            wt_recall = 1.0
            box_contains = True
        elif gt_wt.sum() == 0:
            wt_recall = 1.0 if wt_coarse.sum() == 0 else 0.0
            box_contains = True
        else:
            wt_recall = float((wt_coarse & gt_wt).sum() / (gt_wt.sum() + 1e-8))
            if bb is None:
                box_contains = False
            else:
                lo, hi = bb
                gt_bb = bbox_of(gt_wt)
                box_contains = bool(gt_bb is not None and (lo <= gt_bb[0]).all() and (hi >= gt_bb[1]).all())

        if bb is None:
            s1_empty_count += 1
        s1_recalls.append(wt_recall)
        s1_contains.append(box_contains)

        r = {
            "case": c,
            **{f"{n}_Dice": dice(pred[i], gt[i]) for i, n in enumerate(["WT", "TC", "ET"])},
            **{f"{n}_HD95": compute_hd95(pred[i], gt[i]) for i, n in enumerate(["WT", "TC", "ET"])},
            "S1_Recall": wt_recall,
            "S1_BoxContainsGT": box_contains,
            "S1_FoundNothing": (bb is None),
            "Time_sec": t_elapsed,
            "PeakMem_MB": peak_mb,
        }
        rows.append(r)
        print(f"{c:12s}  WT: {r['WT_Dice']:.3f} (HD: {r['WT_HD95']:.1f})  "
              f"TC: {r['TC_Dice']:.3f} (HD: {r['TC_HD95']:.1f})  "
              f"ET: {r['ET_Dice']:.3f} (HD: {r['ET_HD95']:.1f})  "
              f"S1_Rec: {wt_recall:.3f}  BoxOK: {box_contains}  ({t_elapsed:.2f}s, {peak_mb:.1f}MB)")

    metric_keys = ["WT_Dice", "TC_Dice", "ET_Dice", "WT_HD95", "TC_HD95", "ET_HD95", "S1_Recall", "Time_sec"]
    mean_stats = {k: float(np.mean([r[k] for r in rows])) for k in metric_keys}
    std_stats = {k: float(np.std([r[k] for r in rows])) for k in metric_keys}

    print("\n" + "=" * 80)
    print("SUMMARY EVALUATION RESULTS:")
    for k in ["WT_Dice", "TC_Dice", "ET_Dice"]:
        print(f"  {k}: {mean_stats[k]:.4f} +/- {std_stats[k]:.4f}")
    for k in ["WT_HD95", "TC_HD95", "ET_HD95"]:
        print(f"  {k}: {mean_stats[k]:.2f} +/- {std_stats[k]:.2f} mm")
    print(f"  Stage 1 WT Recall: {mean_stats['S1_Recall']:.4f} +/- {std_stats['S1_Recall']:.4f}")
    print(f"  Stage 1 Box Contains GT: {np.mean(s1_contains)*100:.1f}% ({sum(s1_contains)}/{len(s1_contains)})")
    print(f"  Stage 1 Found Nothing: {s1_empty_count}/{len(cases)} cases")
    print(f"  Average Time/case: {mean_stats['Time_sec']:.2f}s")
    print(f"  Max Peak Memory: {max([r['PeakMem_MB'] for r in rows]):.1f} MB")
    print("=" * 80 + "\n")

    fieldnames = list(rows[0].keys())
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
