"""Generate a synthetic BraTS-like dataset in .npy cache format for dry runs.
Includes:
- Cases with normal central tumors (labels 1, 2, 4)
- Tumor-free case
- Tumor near border
- BraTS standard dimensions (240, 240, 155) or compact dimensions for fast testing
"""
import argparse
import os
import numpy as np


def make_case(shape=(240, 240, 155), tumor_type="normal", rng=None):
    if rng is None:
        rng = np.random.default_rng()

    H, W, D = shape
    # 1. Create brain mask (ellipsoid)
    grid_z, grid_y, grid_x = np.ogrid[:H, :W, :D]
    cz, cy, cx = H // 2, W // 2, D // 2
    rz, ry, rx = H * 0.4, W * 0.4, D * 0.4
    brain_mask = ((grid_z - cz) ** 2 / rz ** 2 +
                  (grid_y - cy) ** 2 / ry ** 2 +
                  (grid_x - cx) ** 2 / rx ** 2) <= 1.0

    # 2. Modalities: 4 channels
    # [flair, t1, t1ce, t2]
    img = np.zeros((4, H, W, D), dtype=np.float32)
    for m in range(4):
        noise = rng.normal(loc=100.0, scale=20.0, size=(H, W, D)).astype(np.float32)
        v = np.where(brain_mask, noise, 0.0)
        # z-score over brain voxels
        nz = v > 0
        if nz.any():
            mu, sd = v[nz].mean(), v[nz].std() + 1e-8
            v = np.where(nz, (v - mu) / sd, 0.0)
        img[m] = v

    lab = np.zeros((H, W, D), dtype=np.uint8)

    if tumor_type == "tumor_free":
        return img.astype(np.float16), lab

    # Determine tumor center and radius
    if tumor_type == "border":
        # near the edge of the brain mask
        t_cz, t_cy, t_cx = cz + int(rz * 0.75), cy + int(ry * 0.75), cx
        t_r = 15
    else:  # normal
        t_cz = cz + rng.integers(-15, 15)
        t_cy = cy + rng.integers(-15, 15)
        t_cx = cx + rng.integers(-15, 15)
        t_r = 20

    dist_sq = (grid_z - t_cz) ** 2 + (grid_y - t_cy) ** 2 + (grid_x - t_cx) ** 2
    # Whole tumor: dist <= t_r
    wt_mask = (dist_sq <= t_r ** 2) & brain_mask
    # Edema (label 2): outer ring
    edema_mask = wt_mask & (dist_sq > (t_r * 0.6) ** 2)
    # Tumor core: dist <= t_r * 0.6
    core_mask = wt_mask & (dist_sq <= (t_r * 0.6) ** 2)
    # Enhancing tumor (label 4): ring of core
    et_mask = core_mask & (dist_sq > (t_r * 0.3) ** 2)
    # Necrotic core (label 1): center of core
    necrotic_mask = core_mask & (dist_sq <= (t_r * 0.3) ** 2)

    lab[edema_mask] = 2
    lab[et_mask] = 4
    lab[necrotic_mask] = 1

    # Signal modulation for tumor regions in image modalities
    # FLAIR bright on edema & WT, T1ce bright on ET, T1 dark on necrosis
    img[0][wt_mask] += 1.5   # FLAIR hyperintense
    img[2][et_mask] += 2.0   # T1ce enhancing
    img[1][necrotic_mask] -= 1.0  # T1 hypointense

    # Re-normalize channels over brain mask
    for m in range(4):
        v = img[m]
        nz = brain_mask
        if nz.any():
            mu, sd = v[nz].mean(), v[nz].std() + 1e-8
            img[m] = np.where(nz, (v - mu) / sd, 0.0)

    return img.astype(np.float16), lab


def generate_dataset(out_dir, n_cases=8, shape=(240, 240, 155), seed=42):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    for i in range(n_cases):
        case_id = f"SYNTH_{i+1:03d}"
        if i == 0:
            tt = "tumor_free"
        elif i == 1:
            tt = "border"
        else:
            tt = "normal"

        img, lab = make_case(shape=shape, tumor_type=tt, rng=rng)
        np.save(os.path.join(out_dir, f"{case_id}_img.npy"), img)
        np.save(os.path.join(out_dir, f"{case_id}_lab.npy"), lab)
        print(f"Generated {case_id}: tumor={tt}, WT voxels={(lab > 0).sum()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_cases", type=int, default=8)
    ap.add_argument("--shape", type=int, nargs=3, default=[240, 240, 155])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    generate_dataset(args.out, n_cases=args.n_cases, shape=tuple(args.shape), seed=args.seed)
