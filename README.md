# Coarse-to-fine Attention-Dilated ResNet U-Net + Transformer bottleneck

Pipeline
1. `python data.py preprocess --raw <BraTS_root> --out cache`   (z-score, save .npy)
2. `python train.py --stage 1 --cache cache --out runs/s1`      (CoarseNet, WT at 1/2 res)
3. `python train.py --stage 2 --cache cache --out runs/s2`      (FineNet on tumour crops)
4. `python infer.py --cache cache --s1 runs/s1/best.pt --s2 runs/s2/best.pt --split test`

Ablations to run for the report (flags already in place)
- `--skip_attn cbam | gate | none`
- `--tf_layers 0..2` (0 would need a tiny edit: skip the bottleneck)
- `--moddrop 0 | 0.3` and `infer.py --drop_mods t1,t2` (missing-modality robustness)
- loss weights in losses.py (boundary term on/off)
- cross-dataset: train on BraTS2020, test on BraTS2021 or another cohort (`--et_label 3` for BraTS2023)

NOTE: written without GPU/data access; only syntax-checked. Run `python models.py` first,
then a 2-epoch dry run on a handful of cases.
