"""BraTS-style data handling.

Raw layout (BraTS2020/2021):  <root>/<case>/<case>_{flair,t1,t1ce,t2,seg}.nii(.gz)

Step 0 (once):  python data.py preprocess --raw <root> --out <cache>
  -> per case:  <case>_img.npy  float16 (4,H,W,D), z-scored over brain voxels, background = 0
                <case>_lab.npy  uint8   (H,W,D)    raw labels
Datasets then read the .npy files (memory-mapped), which is far faster than .nii.gz.
"""
import argparse
import glob
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

MODS = ["flair", "t1", "t1ce", "t2"]


# --------------------------------------------------------------------------- preprocessing
def _find(case_dir, case, suffix):
    hits = glob.glob(os.path.join(case_dir, f"{case}_{suffix}.nii*"))
    return hits[0] if hits else None


def preprocess(raw_root, out_dir):
    import nibabel as nib
    os.makedirs(out_dir, exist_ok=True)
    cases = sorted(d for d in os.listdir(raw_root) if os.path.isdir(os.path.join(raw_root, d)))
    for case in cases:
        cd = os.path.join(raw_root, case)
        chans = []
        for m in MODS:
            p = _find(cd, case, m)
            if p is None:
                break
            v = np.asarray(nib.load(p).dataobj, dtype=np.float32)
            nz = v > 0
            if nz.any():
                mu, sd = v[nz].mean(), v[nz].std() + 1e-8
                v = np.where(nz, (v - mu) / sd, 0.0)
            chans.append(v)
        else:
            np.save(os.path.join(out_dir, f"{case}_img.npy"), np.stack(chans).astype(np.float16))
            sp = _find(cd, case, "seg")
            if sp is not None:
                np.save(os.path.join(out_dir, f"{case}_lab.npy"),
                        np.asarray(nib.load(sp).dataobj).astype(np.uint8))
            print("done", case)
            continue
        print("skip (missing modality)", case)


def list_cases(cache_dir):
    return sorted(os.path.basename(p)[:-8] for p in glob.glob(os.path.join(cache_dir, "*_img.npy")))


def split_cases(cases, val_frac=0.15, test_frac=0.15, seed=0):
    """Patient-wise split (one directory == one patient), so no patient leaks across splits."""
    rng = random.Random(seed)
    cases = cases[:]
    rng.shuffle(cases)
    n_val, n_test = int(len(cases) * val_frac), int(len(cases) * test_frac)
    return cases[n_val + n_test:], cases[:n_val], cases[n_val:n_val + n_test]


# --------------------------------------------------------------------------- label helpers
def to_regions(lab, et_label=4):
    """raw label volume -> (3,...) float masks: WT, TC, ET."""
    lab = np.asarray(lab)
    wt = lab > 0
    et = lab == et_label
    tc = (lab == 1) | et
    return np.stack([wt, tc, et]).astype(np.float32)


def bbox_of(mask, margin=0, min_size=0):
    """Bounding box (start, stop) of a boolean 3D mask with margin; None if empty."""
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return None
    lo, hi = idx.min(0), idx.max(0) + 1
    lo = np.maximum(lo - margin, 0)
    hi = np.minimum(hi + margin, mask.shape)
    # enforce a minimum size, growing around the centre
    for a in range(3):
        if hi[a] - lo[a] < min_size:
            c = (lo[a] + hi[a]) // 2
            lo[a] = c - min_size // 2
            hi[a] = lo[a] + min_size
            if lo[a] < 0:
                hi[a] = min(mask.shape[a], hi[a] - lo[a])
                lo[a] = 0
            elif hi[a] > mask.shape[a]:
                lo[a] = max(0, lo[a] - (hi[a] - mask.shape[a]))
                hi[a] = mask.shape[a]
    return lo, hi


def crop_pad(arr, start, size):
    """Crop arr[..., start:start+size] on the last 3 dims, zero-padding outside the volume."""
    out = np.zeros(arr.shape[:-3] + tuple(size), dtype=arr.dtype)
    src_lo = [max(s, 0) for s in start]
    src_hi = [min(s + z, d) for s, z, d in zip(start, size, arr.shape[-3:])]
    dst_lo = [sl - s for sl, s in zip(src_lo, start)]
    dst_hi = [dl + (sh - sl) for dl, sh, sl in zip(dst_lo, src_hi, src_lo)]
    out[(...,) + tuple(slice(a, b) for a, b in zip(dst_lo, dst_hi))] = \
        arr[(...,) + tuple(slice(a, b) for a, b in zip(src_lo, src_hi))]
    return out


def random_flip(*arrs):
    for ax in (-3, -2, -1):
        if random.random() < 0.5:
            arrs = tuple(np.flip(a, ax) for a in arrs)
    return tuple(np.ascontiguousarray(a) for a in arrs)


# --------------------------------------------------------------------------- datasets
class CoarseDataset(Dataset):
    """Stage 1: whole volume downsampled by `down` (default 2) -> WT mask."""

    def __init__(self, cache, cases, down=2, train=True, et_label=4):
        self.cache, self.cases, self.down, self.train, self.et = cache, cases, down, train, et_label

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        c = self.cases[i]
        img = np.load(os.path.join(self.cache, f"{c}_img.npy"), mmap_mode="r")
        lab = np.load(os.path.join(self.cache, f"{c}_lab.npy"), mmap_mode="r")
        wt = (np.asarray(lab) > 0).astype(np.float32)[None]
        x = torch.from_numpy(np.asarray(img, dtype=np.float32))[None]
        y = torch.from_numpy(wt)[None]
        x = F.avg_pool3d(x, self.down)[0]
        y = F.max_pool3d(y, self.down)[0]
        x, y = x.numpy(), y.numpy()
        if self.train:
            x, y = random_flip(x, y)
        return torch.from_numpy(x), torch.from_numpy(y)


class FineDataset(Dataset):
    """Stage 2: fixed patch around the GT whole tumour, with random margin/centre jitter
    (simulates imperfect stage-1 crops)."""

    def __init__(self, cache, cases, patch=128, jitter=16, train=True, et_label=4):
        self.cache, self.cases, self.patch, self.jitter = cache, cases, patch, jitter
        self.train, self.et = train, et_label

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        c = self.cases[i]
        img = np.load(os.path.join(self.cache, f"{c}_img.npy"), mmap_mode="r")
        lab = np.load(os.path.join(self.cache, f"{c}_lab.npy"), mmap_mode="r")
        lab = np.asarray(lab)
        bb = bbox_of(lab > 0)
        if bb is None:                                    # tumour-free case: random centre
            centre = np.array([s // 2 for s in lab.shape])
        else:
            centre = (bb[0] + bb[1]) // 2
        if self.train:
            centre = centre + np.random.randint(-self.jitter, self.jitter + 1, size=3)
        start = centre - self.patch // 2
        size = (self.patch,) * 3
        x = crop_pad(np.asarray(img, dtype=np.float32), start, size)
        y = to_regions(crop_pad(lab, start, size), self.et)   # crop first: cheap
        if self.train:
            x, y = random_flip(x, y)
        return torch.from_numpy(x), torch.from_numpy(y)


# --------------------------------------------------------------------------- augmentation
def modality_dropout(x, p=0.3):
    """With probability p per sample, zero a random subset of modalities (keeps >=1).
    Inputs are z-scored, so 0 == 'mean intensity' == 'missing'."""
    b, c = x.shape[:2]
    keep = (torch.rand(b, c, device=x.device) > 0.5)
    empty = keep.sum(1) == 0
    if empty.any():
        keep[empty, torch.randint(0, c, (int(empty.sum()),), device=x.device)] = True
    apply = torch.rand(b, device=x.device) < p
    keep = torch.where(apply[:, None], keep, torch.ones_like(keep))
    return x * keep.view(b, c, 1, 1, 1).to(x.dtype)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["preprocess"])
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    preprocess(a.raw, a.out)
