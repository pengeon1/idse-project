"""Master Kaggle Execution Pipeline for BraTS 3D Coarse-to-Fine Segmentation.
Runs the entire Phase B, C, D pipeline end-to-end:
1. Preprocess raw BraTS dataset to /kaggle/working/cache
2. Dataset spot-check and verification montage
3. Model parameter & memory profiling
4. Train Stage 1 (CoarseNet)
5. Train Stage 2 (FineNet - A0 Full Model)
6. Evaluate full two-stage pipeline on test set (Dice, HD95, Stage 1 diagnostics, timing, memory)
7. Run complete ablation suite (A1 to A7) + missing modality tests
8. Generate plots, qualitative PNG overlays, markdown summary table, and results zip bundle.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

def run_cmd(cmd, desc=None):
    if desc:
        print(f"\n{'='*70}\n>>> {desc}\n{'='*70}")
    print(f"Executing: {cmd}")
    t0 = time.time()
    res = subprocess.run(cmd, shell=True, check=True)
    print(f"[Done in {time.time()-t0:.1f}s]")
    return res

def locate_brats_data(search_roots=None):
    if search_roots is None:
        search_roots = ["/kaggle/input", "./data", "."]
    print(f"Searching for raw BraTS dataset in: {search_roots}")
    if os.path.exists("/kaggle/input"):
        print("Mounted folders in /kaggle/input:")
        for item in sorted(os.listdir("/kaggle/input")):
            print(f"  - /kaggle/input/{item}")

    for root in search_roots:
        if not os.path.exists(root):
            continue
        # Search for any patient folder with flair
        for root_dir, dirs, files in os.walk(root):
            flair_files = [f for f in files if "flair" in f.lower() and (".nii" in f.lower() or ".mha" in f.lower())]
            if flair_files:
                parent = os.path.dirname(root_dir)
                print(f"\n[+] Found BraTS flair sample: {os.path.join(root_dir, flair_files[0])}")
                print(f"[+] Patient directory: {root_dir}")
                print(f"[+] Identified dataset root: {parent}\n")
                return parent

    raise FileNotFoundError(
        "Could not locate any BraTS NIfTI files (*flair*.nii*) in /kaggle/input.\n"
        "Please check if you added a BraTS dataset via '+ Add Input' on the top right of your Kaggle notebook!"
    )

def spot_check_dataset(cache_dir, out_dir="results/spot_check"):
    os.makedirs(out_dir, exist_ok=True)
    img_files = sorted(glob.glob(os.path.join(cache_dir, "*_img.npy")))[:3]
    print(f"\n--- Spot Checking 3 Preprocessed Cases ---")
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for row_idx, f in enumerate(img_files):
        cid = os.path.basename(f)[:-8]
        img = np.load(f)
        lab = np.load(os.path.join(cache_dir, f"{cid}_lab.npy"))
        
        print(f"\nCase: {cid}")
        print(f"  Image shape: {img.shape}, dtype: {img.dtype}")
        print(f"  Label shape: {lab.shape}, dtype: {lab.dtype}, unique labels: {np.unique(lab)}")
        
        # Verify stats of brain voxels (img != 0)
        for m_idx, m_name in enumerate(["FLAIR", "T1", "T1ce", "T2"]):
            m_data = img[m_idx]
            nz = m_data != 0
            if nz.any():
                print(f"  {m_name:6s} -> non-zero mean: {m_data[nz].mean():.4f}, std: {m_data[nz].std():.4f}")
            else:
                print(f"  {m_name:6s} -> all zero!")
                
        # Mid-slice with tumor
        wt_sum = (lab > 0).sum(axis=(0, 1))
        mid_z = int(np.argmax(wt_sum)) if wt_sum.max() > 0 else lab.shape[-1] // 2
        
        flair = img[0, :, :, mid_z]
        t1ce = img[2, :, :, mid_z]
        wt_mask = (lab[:, :, mid_z] > 0).astype(float)
        
        axes[row_idx, 0].imshow(flair, cmap="gray")
        axes[row_idx, 0].set_title(f"{cid} FLAIR (z={mid_z})")
        axes[row_idx, 0].axis("off")
        
        axes[row_idx, 1].imshow(t1ce, cmap="gray")
        axes[row_idx, 1].set_title(f"{cid} T1ce (z={mid_z})")
        axes[row_idx, 1].axis("off")
        
        axes[row_idx, 2].imshow(lab[:, :, mid_z], cmap="nipy_spectral", interpolation="nearest")
        axes[row_idx, 2].set_title(f"Ground Truth Labels")
        axes[row_idx, 2].axis("off")
        
        axes[row_idx, 3].imshow(flair, cmap="gray")
        axes[row_idx, 3].imshow(wt_mask, cmap="autumn", alpha=0.4)
        axes[row_idx, 3].set_title(f"FLAIR + WT Mask Overlay")
        axes[row_idx, 3].axis("off")

    plt.tight_layout()
    montage_path = os.path.join(out_dir, "spot_check_montage.png")
    plt.savefig(montage_path, dpi=200)
    plt.close()
    print(f"\nSpot check montage saved to: {montage_path}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default=None, help="Path to raw BraTS folder")
    parser.add_argument("--work_dir", default="/kaggle/working", help="Working directory")
    parser.add_argument("--s1_epochs", type=int, default=100)
    parser.add_argument("--s2_epochs", type=int, default=150)
    parser.add_argument("--ablation_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--base_fine", type=int, default=16)
    parser.add_argument("--base_coarse", type=int, default=8)
    parser.add_argument("--skip_ablations", action="store_true")
    args = parser.parse_args()

    work_dir = os.path.abspath(args.work_dir)
    cache_dir = os.path.join(work_dir, "cache")
    runs_dir = os.path.join(work_dir, "runs")
    results_dir = os.path.join(work_dir, "results")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Locate dataset
    if args.raw_dir and os.path.exists(args.raw_dir):
        raw_dir = args.raw_dir
    else:
        if args.raw_dir:
            print(f"Notice: Path passed to --raw_dir ('{args.raw_dir}') does not exist.")
            print("Auto-discovering BraTS dataset under /kaggle/input...")
        raw_dir = locate_brats_data()

    # 2. Preprocessing
    if not os.path.exists(cache_dir) or len(glob.glob(os.path.join(cache_dir, "*_img.npy"))) == 0:
        run_cmd(f"python data.py preprocess --raw '{raw_dir}' --out '{cache_dir}'",
                desc="Step 1: Preprocessing raw BraTS NIfTI files to z-scored .npy cache")
    else:
        print(f"Found existing preprocessed cache in {cache_dir}")

    # 3. Spot check
    spot_check_dataset(cache_dir, out_dir=os.path.join(results_dir, "spot_check"))

    # 4. Train Stage 1 (CoarseNet)
    s1_out = os.path.join(runs_dir, "s1")
    if not os.path.exists(os.path.join(s1_out, "best.pt")):
        run_cmd(f"python train.py --stage 1 --cache '{cache_dir}' --out '{s1_out}' "
                f"--epochs {args.s1_epochs} --bs {args.batch_size} --base {args.base_coarse} --workers 4",
                desc=f"Step 2: Training Stage 1 CoarseNet ({args.s1_epochs} epochs)")
        run_cmd(f"python scripts/visualize_results.py --history '{s1_out}/history.csv' "
                f"--curve_out '{results_dir}/curves_stage1.png'", desc="Plot Stage 1 training curves")

    # 5. Train Stage 2 (FineNet - A0 Full Model)
    s2_out = os.path.join(runs_dir, "s2_a0_full")
    if not os.path.exists(os.path.join(s2_out, "best.pt")):
        run_cmd(f"python train.py --stage 2 --cache '{cache_dir}' --out '{s2_out}' "
                f"--epochs {args.s2_epochs} --bs {args.batch_size} --base {args.base_fine} "
                f"--skip_attn cbam --tf_layers 2 --moddrop 0.3 --w_bd 0.5 --workers 4",
                desc=f"Step 3: Training Stage 2 FineNet A0 Full Model ({args.s2_epochs} epochs)")
        run_cmd(f"python scripts/visualize_results.py --history '{s2_out}/history.csv' "
                f"--curve_out '{results_dir}/curves_stage2_a0.png'", desc="Plot Stage 2 A0 training curves")

    # 6. Evaluate A0 on held-out test split
    a0_csv = os.path.join(results_dir, "eval_a0_full.csv")
    run_cmd(f"python infer.py --cache '{cache_dir}' --s1 '{s1_out}/best.pt' --s2 '{s2_out}/best.pt' "
            f"--split test --out_csv '{a0_csv}'",
            desc="Step 4: Full two-stage pipeline evaluation on held-out test split")

    # Generate qualitative visual overlays for A0
    run_cmd(f"python scripts/visualize_results.py --csv '{a0_csv}' --cache '{cache_dir}' "
            f"--s1 '{s1_out}/best.pt' --s2 '{s2_out}/best.pt' --out_dir '{results_dir}/overlays_a0'",
            desc="Generate qualitative 4-panel overlays (best, median, worst, S1 failure)")

    if args.skip_ablations:
        print("\nSkipping ablations as requested.")
        return

    # 7. Ablation Suite (A1 - A7)
    ablations = [
        {"id": "A1", "desc": "Attention Gate skip", "flags": "--skip_attn gate --tf_layers 2 --moddrop 0.3 --w_bd 0.5"},
        {"id": "A2", "desc": "No skip attention", "flags": "--skip_attn none --tf_layers 2 --moddrop 0.3 --w_bd 0.5"},
        {"id": "A3", "desc": "No Transformer bottleneck", "flags": "--skip_attn cbam --tf_layers 0 --moddrop 0.3 --w_bd 0.5"},
        {"id": "A4", "desc": "No dilated convolutions", "flags": "--skip_attn cbam --tf_layers 2 --no_dilated --moddrop 0.3 --w_bd 0.5"},
        {"id": "A5", "desc": "No modality dropout", "flags": "--skip_attn cbam --tf_layers 2 --moddrop 0.0 --w_bd 0.5"},
        {"id": "A6", "desc": "No boundary loss", "flags": "--skip_attn cbam --tf_layers 2 --moddrop 0.3 --w_bd 0.0"},
    ]

    ablation_results = {}
    
    # Read A0 summary from CSV
    import pandas as pd
    df_a0 = pd.read_csv(a0_csv)
    ablation_results["A0 (Full)"] = {
        "WT_Dice": f"{df_a0['WT_Dice'].mean():.4f}±{df_a0['WT_Dice'].std():.3f}",
        "TC_Dice": f"{df_a0['TC_Dice'].mean():.4f}±{df_a0['TC_Dice'].std():.3f}",
        "ET_Dice": f"{df_a0['ET_Dice'].mean():.4f}±{df_a0['ET_Dice'].std():.3f}",
        "WT_HD95": f"{df_a0['WT_HD95'].mean():.2f}",
        "Time/case": f"{df_a0['Time_sec'].mean():.2f}s",
        "Max_Mem": f"{df_a0['PeakMem_MB'].max():.0f}MB"
    }

    # Run A1 - A6
    for ab in ablations:
        aid, aname, aflags = ab["id"], ab["desc"], ab["flags"]
        out_a = os.path.join(runs_dir, f"s2_{aid.lower()}")
        csv_a = os.path.join(results_dir, f"eval_{aid.lower()}.csv")
        
        if not os.path.exists(os.path.join(out_a, "best.pt")):
            run_cmd(f"python train.py --stage 2 --cache '{cache_dir}' --out '{out_a}' "
                    f"--epochs {args.ablation_epochs} --bs {args.batch_size} --base {args.base_fine} "
                    f"{aflags} --workers 4", desc=f"Ablation {aid}: {aname} ({args.ablation_epochs} epochs)")
            
        run_cmd(f"python infer.py --cache '{cache_dir}' --s1 '{s1_out}/best.pt' --s2 '{out_a}/best.pt' "
                f"--split test --out_csv '{csv_a}'", desc=f"Evaluating Ablation {aid} ({aname})")
        
        df_a = pd.read_csv(csv_a)
        ablation_results[f"{aid} ({aname})"] = {
            "WT_Dice": f"{df_a['WT_Dice'].mean():.4f}±{df_a['WT_Dice'].std():.3f}",
            "TC_Dice": f"{df_a['TC_Dice'].mean():.4f}±{df_a['TC_Dice'].std():.3f}",
            "ET_Dice": f"{df_a['ET_Dice'].mean():.4f}±{df_a['ET_Dice'].std():.3f}",
            "WT_HD95": f"{df_a['WT_HD95'].mean():.2f}",
            "Time/case": f"{df_a['Time_sec'].mean():.2f}s",
            "Max_Mem": f"{df_a['PeakMem_MB'].max():.0f}MB"
        }

    # A7: Single-Stage Baseline (FineNet directly on full volume without Stage 1)
    a7_csv = os.path.join(results_dir, "eval_a7_single_stage.csv")
    run_cmd(f"python infer.py --cache '{cache_dir}' --s2 '{s2_out}/best.pt' "
            f"--split test --single_stage --out_csv '{a7_csv}'",
            desc="Ablation A7: Single-stage baseline (FineNet directly on whole volume)")
    df_a7 = pd.read_csv(a7_csv)
    ablation_results["A7 (Single-stage)"] = {
        "WT_Dice": f"{df_a7['WT_Dice'].mean():.4f}±{df_a7['WT_Dice'].std():.3f}",
        "TC_Dice": f"{df_a7['TC_Dice'].mean():.4f}±{df_a7['TC_Dice'].std():.3f}",
        "ET_Dice": f"{df_a7['ET_Dice'].mean():.4f}±{df_a7['ET_Dice'].std():.3f}",
        "WT_HD95": f"{df_a7['WT_HD95'].mean():.2f}",
        "Time/case": f"{df_a7['Time_sec'].mean():.2f}s",
        "Max_Mem": f"{df_a7['PeakMem_MB'].max():.0f}MB"
    }

    # Missing Modality Tests on A0 (with modality dropout) vs A5 (without modality dropout)
    print("\n--- Missing Modality Robustness Evaluation ---")
    missing_tests = ["t1,t2", "flair", "t1ce"]
    missing_results = []
    for mods in missing_tests:
        for model_id, model_dir in [("A0_ModDrop", s2_out), ("A5_NoModDrop", os.path.join(runs_dir, "s2_a5"))]:
            csv_m = os.path.join(results_dir, f"eval_missing_{model_id}_{mods.replace(',', '_')}.csv")
            run_cmd(f"python infer.py --cache '{cache_dir}' --s1 '{s1_out}/best.pt' --s2 '{model_dir}/best.pt' "
                    f"--split test --drop_mods '{mods}' --out_csv '{csv_m}'",
                    desc=f"Testing {model_id} with missing sequence: {mods}")
            df_m = pd.read_csv(csv_m)
            missing_results.append({
                "Model": model_id,
                "Dropped": mods,
                "WT_Dice": f"{df_m['WT_Dice'].mean():.4f}",
                "TC_Dice": f"{df_m['TC_Dice'].mean():.4f}",
                "ET_Dice": f"{df_m['ET_Dice'].mean():.4f}",
            })

    # 8. Output Markdown Tables & Summary
    df_table = pd.DataFrame.from_dict(ablation_results, orient="index")
    md_table = df_table.to_markdown()
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY TABLE:")
    print("="*80)
    print(md_table)
    
    df_missing = pd.DataFrame(missing_results)
    print("\nMISSING MODALITY ROBUSTNESS TABLE:")
    print(df_missing.to_markdown(index=False))

    with open(os.path.join(results_dir, "ABLATION_TABLE.md"), "w") as f:
        f.write("# Ablation Study Results\n\n" + md_table + "\n\n## Missing Modality Robustness\n\n" + df_missing.to_markdown(index=False) + "\n")

    # 9. Create downloadable ZIP archive
    zip_path = os.path.join(work_dir, "results_bundle.zip")
    shutil.make_archive(zip_path.replace(".zip", ""), 'zip', results_dir)
    print(f"\nAll results, curves, overlays, and tables successfully bundled to: {zip_path}")
    print("Kaggle pipeline execution completed successfully!")

if __name__ == "__main__":
    main()
