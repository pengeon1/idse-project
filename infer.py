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


@torch.no_grad()
def segment(coarse, fine, img, dev, down=2, margin=12, thr=0.5, return_bbox=False):
    """img: float array (4,H,W,D), already z-scored. Returns bool (3,H,W,D): WT, TC, ET."""
    x = torch.from_numpy(np.asarray(img, dtype=np.float32))[None].to(dev)
    shape = x.shape[2:]
    use_amp = dev == "cuda"

    # ---- stage 1: coarse whole-tumour localisation at low resolution
    xl, sz = pad_to_multiple(F.avg_pool3d(x, down), 8)
    with torch.autocast(dev, dtype=torch.float16, enabled=use_amp):
        p = unpad(torch.sigmoid(coarse(xl)), sz)
    p = F.interpolate(p.float(), size=shape, mode="trilinear", align_corners=False)
    wt_coarse = largest_component((p[0, 0] > thr).cpu().numpy())
    bb = bbox_of(wt_coarse, margin=margin, min_size=64)

    out = np.zeros((3,) + tuple(shape), dtype=bool)
    if bb is None:                                        # nothing found
        return (out, None) if return_bbox else out
    lo, hi = bb

    # ---- stage 2: fine segmentation of the crop
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    crop = x[(slice(None), slice(None)) + sl]
    crop_p, csz = pad_to_multiple(crop, 16)
    with torch.autocast(dev, dtype=torch.float16, enabled=use_amp):
        prob = unpad(torch.sigmoid(fine(crop_p)), csz)[0].float().cpu().numpy() > thr
    wt, tc, et = prob
    tc &= wt                                              # enforce ET <= TC <= WT
    et &= tc
    out[(slice(None),) + sl] = np.stack([wt, tc, et])
    return (out, (lo, hi)) if return_bbox else out


def dice(a, b, eps=1e-5):
    a, b = a.astype(bool), b.astype(bool)
    if a.sum() + b.sum() == 0:
        return 1.0
    return float(2 * (a & b).sum() / (a.sum() + b.sum() + eps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--s1", required=True)
    ap.add_argument("--s2", required=True)
    ap.add_argument("--split_file", default=None, help="split.json from training (default: next to --s2)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--et_label", type=int, default=4)
    ap.add_argument("--margin", type=int, default=12)
    ap.add_argument("--drop_mods", default="", help="comma list of modalities to zero out, e.g. t1,t2")
    ap.add_argument("--out_csv", default="results.csv")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    coarse, fine = load_models(a.s1, a.s2, dev)
    split_file = a.split_file or os.path.join(os.path.dirname(a.s2), "split.json")
    cases = json.load(open(split_file))[a.split]
    drop = [MODS.index(m) for m in a.drop_mods.split(",") if m]

    rows = []
    for c in cases:
        img = np.load(os.path.join(a.cache, f"{c}_img.npy")).astype(np.float32)
        lab = np.load(os.path.join(a.cache, f"{c}_lab.npy"))
        for i in drop:
            img[i] = 0
        pred = segment(coarse, fine, img, dev, margin=a.margin)
        gt = to_regions(lab, a.et_label) > 0.5
        r = {"case": c, **{n: dice(pred[i], gt[i]) for i, n in enumerate(["WT", "TC", "ET"])}}
        rows.append(r)
        print(c, " ".join(f"{k} {v:.3f}" for k, v in r.items() if k != "case"))

    mean = {k: float(np.mean([r[k] for r in rows])) for k in ["WT", "TC", "ET"]}
    print("MEAN", " ".join(f"{k} {v:.4f}" for k, v in mean.items()))
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "WT", "TC", "ET"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
