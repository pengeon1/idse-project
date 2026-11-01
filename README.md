# Coarse-to-Fine 3D Brain Tumor Segmentation
**Attention-Dilated ResNet U-Net with a Transformer Bottleneck**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6%2Bcu124-orange.svg)](https://pytorch.org/)
[![Smoke Tests](https://img.shields.io/badge/tests-7%2F7%20passed-brightgreen.svg)](tests/test_smoke.py)
[![BraTS 2020](https://img.shields.io/badge/dataset-BraTS%202020-blueviolet.svg)](https://www.med.upenn.edu/cbica/brats2020/)

**Course:** CS3006 Introduction to Data Science for Engineers (Dr. Jagadeesh Kakarla), Team 12, IIITDM Kancheepuram.  
**Authors:** Narayana Murthy S, Piyush Mishra, Ankush R, Vihaan Gupta.

---

## Architecture Overview

A lightweight, two-stage coarse-to-fine 3D segmentation pipeline combining techniques from three surveyed papers:
1. **Zhu et al. 2024:** ResNet encoder blocks with CBAM (Convolutional Block Attention Module) skip attention and composite Dice + BCE loss.
2. **Guan et al. 2023:** 3D Kernel-sharing dilated convolutions in deep stages 3 and 4 for multi-scale receptive field expansion without parameter growth.
3. **Zhang et al. 2024:** Transformer bottleneck (2 layers) with conditional depthwise positional encoding for global context at 1/16 resolution.

### Two-Stage Pipeline Flow
1. **Stage 1 (`CoarseNet`):** Light 3D U-Net (base 8 channels, 0.36M params) operates on the full volume downsampled 2× ($120 \times 120 \times 80$) to predict Whole Tumor (WT). At inference, probabilities are upsampled, thresholded at 0.5, filtered for the largest connected component, and expanded with a 12-voxel safety margin (min crop size 64).
2. **Stage 2 (`FineNet`):** Full-resolution multi-label network (base 16 channels, 9.27M params) segments the localized 128³ patch into Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET), strictly enforcing $ET \le TC \le WT$.

---

## Key Performance Results

Evaluated on the held-out test split of BraTS 2020 ($N=27$ patients):

| Metric | Whole Tumor (WT) | Tumor Core (TC) | Enhancing Tumor (ET) |
|:---|:---:|:---:|:---:|
| **Dice Score (Mean ± Std)** | **0.9127 ± 0.065** | **0.8994 ± 0.112** | **0.8488 ± 0.088** |
| **Median Dice** | **0.9321** | **0.9349** | **0.8663** |
| **95% Hausdorff Distance** | **2.76 mm** | **2.82 mm** | **1.53 mm** |
| **Inference Latency** | \multicolumn{3}{c|}{**0.388 seconds / volume** (2.58 volumes/second)} |
| **Peak GPU VRAM** | \multicolumn{3}{c|}{**818.1 MB** (FP16 mixed precision)} |
| **Stage 1 Detection Rate** | \multicolumn{3}{c|}{**100% (0 / 27 empty predictions)**, 95.1% WT recall} |

Detailed ablation studies (A0 through A7) and missing modality analyses are documented in [REPORT.md](REPORT.md).

---

## Project Structure

```
cfseg/
├── data.py                   # BraTS preprocessing, memory-mapped caching, Dataset loaders, modality dropout
├── models.py                 # CoarseNet, FineNet, CBAM, Attention Gates, Dilated Conv, TransformerBottleneck
├── losses.py                 # Soft Dice, Binary Cross-Entropy, Soft Boundary-Dice loss
├── train.py                  # OneCycleLR training engine with AMP autocast and history logging
├── infer.py                  # Two-stage & single-stage inference, HD95, latency, peak VRAM profiling
├── requirements.txt          # Minimal pinned dependencies
├── REPORT.md                 # Comprehensive project report & ablation analysis
├── tests/
│   └── test_smoke.py         # 7 unit tests covering forward passes, shapes, gradients, and edge cases
├── scripts/
│   ├── run_kaggle_pipeline.py# Master end-to-end automated runner (preprocessing, training, ablations, zipping)
│   ├── visualize_results.py  # Generates loss/Dice curves and 4-panel qualitative overlays
│   └── make_synthetic_cache.py # Generates synthetic test volumes for offline verification
└── results/                  # Ablation tables, CSVs, training curves, and qualitative PNG overlays
```

---

## Quickstart & Reproducibility

### 1. Installation
```bash
git clone https://github.com/PiyushMishra12/cfseg.git
cd cfseg
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify Unit Tests
```bash
python -m pytest tests/test_smoke.py -v
```

### 3. Preprocess Raw BraTS Data
```bash
python data.py preprocess \
    --raw /path/to/MICCAI_BraTS2020_TrainingData \
    --out cache \
    --max_cases 180
```

### 4. Train Stage 1 (CoarseNet)
```bash
python train.py --stage 1 \
    --cache cache \
    --out runs/s1 \
    --epochs 100 \
    --bs 2 \
    --base 8 \
    --workers 4
```

### 5. Train Stage 2 (FineNet A0 Full Model)
```bash
python train.py --stage 2 \
    --cache cache \
    --out runs/s2_a0_full \
    --epochs 150 \
    --bs 2 \
    --base 16 \
    --skip_attn cbam \
    --tf_layers 2 \
    --moddrop 0.3 \
    --w_bd 0.5 \
    --workers 4
```

### 6. Evaluate Two-Stage Pipeline
```bash
python infer.py \
    --cache cache \
    --s1 runs/s1/best.pt \
    --s2 runs/s2_a0_full/best.pt \
    --split test \
    --out_csv results/eval_a0_full.csv
```

### 7. Run Full Pipeline End-to-End (Kaggle or Local)
```bash
python scripts/run_kaggle_pipeline.py \
    --raw_dir /path/to/MICCAI_BraTS2020_TrainingData \
    --work_dir . \
    --max_cases 180 \
    --s1_epochs 100 \
    --s2_epochs 150 \
    --ablation_epochs 60 \
    --batch_size 2
```

---

## Citation
If you use this codebase or pipeline, please reference:
```bibtex
@misc{cfseg2026,
  title={Coarse-to-Fine 3D Brain Tumor Segmentation with Attention-Dilated ResNet U-Net and Transformer Bottleneck},
  author={Murthy, Narayana and Mishra, Piyush and Ankush, R. and Gupta, Vihaan},
  institution={IIITDM Kancheepuram},
  year={2026}
}
```
