"""Master Evaluation Engine for Advanced Enhancements:
A. Cross-Dataset Generalization & Scanner Shift Stress Test (Gamma, Bias Field, Noise, or BraTS 2021/2023)
B. Uncertainty-Aware Adaptive Margin for Stage 1 (Fixing peripheral edema truncation)
C. 4-Fold Test-Time Augmentation (TTA) on Stage 2 FineNet
"""
import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import numpy as np

def run_cmd(cmd, desc=None):
    if desc:
        print(f"\n{'='*70}\n>>> {desc}\n{'='*70}")
    print(f"Executing: {cmd}")
    t0 = time.time()
    res = subprocess.run(cmd, shell=True, check=True)
    print(f"[Done in {time.time()-t0:.1f}s]")
    return res

def summarize_csv(p):
    with open(p, "r") as f:
        rows = list(csv.DictReader(f))
    def col(k): return [float(r[k]) for r in rows if k in r and r[k] != '']
    wt_d = col("WT_Dice")
    tc_d = col("TC_Dice")
    et_d = col("ET_Dice")
    wt_h = col("WT_HD95")
    s1_rec = col("S1_Recall")
    box_ok = [1 if r.get("S1_BoxContainsGT", "").lower() == "true" else 0 for r in rows]
    t = col("Time_sec")
    mem = col("PeakMem_MB")
    return {
        "WT_Dice": f"{np.mean(wt_d):.4f}±{np.std(wt_d):.3f}" if wt_d else "N/A",
        "TC_Dice": f"{np.mean(tc_d):.4f}±{np.std(tc_d):.3f}" if tc_d else "N/A",
        "ET_Dice": f"{np.mean(et_d):.4f}±{np.std(et_d):.3f}" if et_d else "N/A",
        "WT_HD95": f"{np.mean(wt_h):.2f}" if wt_h else "N/A",
        "S1_Recall": f"{np.mean(s1_rec)*100:.1f}%" if s1_rec else "N/A",
        "Box_OK": f"{sum(box_ok)}/{len(rows)}" if rows else "N/A",
        "Time/case": f"{np.mean(t):.2f}s" if t else "N/A",
        "Peak_Mem": f"{np.max(mem):.0f}MB" if mem else "N/A"
    }

def dict_to_markdown_table(data_dict):
    if not data_dict:
        return ""
    headers = ["Evaluation Scenario", "WT_Dice", "TC_Dice", "ET_Dice", "WT_HD95", "S1_Recall", "Box_OK", "Time/case", "Peak_Mem"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---"] + [":---:"] * (len(headers) - 1)) + " |"
    ]
    for row_name, metrics in data_dict.items():
        row_str = f"| **{row_name}** | " + " | ".join(str(metrics.get(h, "")) for h in headers[1:]) + " |"
        lines.append(row_str)
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", default="/kaggle/working", help="Working directory")
    parser.add_argument("--cache", default=None, help="Cache directory (defaults to work_dir/cache)")
    parser.add_argument("--s1", default=None, help="Path to s1 best.pt")
    parser.add_argument("--s2", default=None, help="Path to s2_a0_full best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--cross_raw", default=None, help="Path to raw second dataset (e.g. BraTS 2021/2023)")
    args = parser.parse_args()

    work_dir = os.path.abspath(args.work_dir)
    cache_dir = args.cache or os.path.join(work_dir, "cache")
    runs_dir = os.path.join(work_dir, "runs")
    results_dir = os.path.join(work_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    s1_path = args.s1 or os.path.join(runs_dir, "s1", "best.pt")
    s2_path = args.s2 or os.path.join(runs_dir, "s2_a0_full", "best.pt")

    if not os.path.exists(s1_path) or not os.path.exists(s2_path):
        raise FileNotFoundError(f"Model checkpoints not found:\n  S1: {s1_path}\n  S2: {s2_path}\nPlease ensure Stage 1 and Stage 2 have been trained.")

    print("\n" + "="*80)
    print("RUNNING ADVANCED IMPROVEMENTS: A (Generalization), B (Adaptive Margin), C (TTA)")
    print("="*80)

    eval_runs = [
        # 1. Baseline
        {"name": "1. Baseline (A0 Full)", "flags": "", "csv": os.path.join(results_dir, "eval_baseline.csv"),
         "desc": "Standard Two-Stage Pipeline (fixed margin 12, single-pass)"},
        # 2. Enhancement B: Adaptive Margin
        {"name": "2. Adaptive Margin (B)", "flags": "--adaptive_margin --box_thr 0.35",
         "csv": os.path.join(results_dir, "eval_adaptive_margin.csv"),
         "desc": "Uncertainty-aware bounding box candidate extraction & dynamic margin expansion"},
        # 3. Enhancement C: Test-Time Augmentation
        {"name": "3. Test-Time Aug (C)", "flags": "--tta",
         "csv": os.path.join(results_dir, "eval_tta.csv"),
         "desc": "4-fold test-time augmentation (axial/coronal/sagittal flip ensembling)"},
        # 4. Combined B + C
        {"name": "4. Combined Adaptive + TTA (B+C)", "flags": "--adaptive_margin --box_thr 0.35 --tta",
         "csv": os.path.join(results_dir, "eval_adaptive_tta.csv"),
         "desc": "Adaptive Margin + 4-fold TTA combined"},
        # 5. Stress Tests A: Scanner Domain Shifts
        {"name": "5. Scanner Shift: Gamma Low", "flags": "--shift_type gamma_low",
         "csv": os.path.join(results_dir, "eval_shift_gamma_low.csv"),
         "desc": "Simulated hyper-intense MRI scanner non-linearity (gamma = 0.75)"},
        {"name": "6. Scanner Shift: Gamma High", "flags": "--shift_type gamma_high",
         "csv": os.path.join(results_dir, "eval_shift_gamma_high.csv"),
         "desc": "Simulated hypo-intense MRI scanner non-linearity (gamma = 1.35)"},
        {"name": "7. Scanner Shift: Bias Field", "flags": "--shift_type bias_field",
         "csv": os.path.join(results_dir, "eval_shift_bias_field.csv"),
         "desc": "Simulated RF coil spatial inhomogeneity / B0 magnetic gradient field"},
        {"name": "8. Scanner Shift: Low SNR Noise", "flags": "--shift_type noise",
         "csv": os.path.join(results_dir, "eval_shift_noise.csv"),
         "desc": "Simulated low-field / noisy acquisition (Gaussian perturbation)"},
    ]

    summary_records = {}

    for item in eval_runs:
        if os.path.exists(item['csv']) and os.path.getsize(item['csv']) > 0:
            print(f"\n[+] Found existing results for {item['name']} at {item['csv']}. Skipping execution.")
            summary_records[item['name']] = summarize_csv(item['csv'])
            continue
        cmd = (f"python infer.py --cache '{cache_dir}' --s1 '{s1_path}' --s2 '{s2_path}' "
               f"--split {args.split} {item['flags']} --out_csv '{item['csv']}'")
        run_cmd(cmd, desc=f"{item['name']}: {item['desc']}")
        summary_records[item['name']] = summarize_csv(item['csv'])

    # Optional: Cross-Dataset Zero-Shot Transfer
    cross_raw = args.cross_raw
    if not cross_raw and os.path.exists("/kaggle/input"):
        # Auto-detect if a 2021 or 2023 dataset was added to /kaggle/input
        for root_dir, dirs, files in os.walk("/kaggle/input"):
            if ("brats2021" in root_dir.lower() or "brats2023" in root_dir.lower() or "brats-2021" in root_dir.lower()) and any("flair" in f.lower() for f in files):
                parent = os.path.dirname(root_dir)
                has_seg = any("seg" in f.lower() for f in files)
                if has_seg:
                    cross_raw = parent
                    break

    if cross_raw and os.path.exists(cross_raw):
        print(f"\n[+] Found secondary cross-dataset at: {cross_raw}")
        cross_cache = os.path.join(work_dir, "cache_cross")
        os.makedirs(cross_cache, exist_ok=True)
        # Preprocess up to 30 cases
        run_cmd(f"python data.py preprocess --raw '{cross_raw}' --out '{cross_cache}' --max_cases 30",
                desc="Preprocessing secondary dataset for zero-shot cross-dataset evaluation")
        cross_csv = os.path.join(results_dir, "eval_cross_dataset.csv")
        # Determine et_label: 3 for BraTS 2023, 4 for BraTS 2021
        et_lbl = 3 if "2023" in cross_raw else 4
        run_cmd(f"python infer.py --cache '{cross_cache}' --s1 '{s1_path}' --s2 '{s2_path}' "
                f"--all_cases --et_label {et_lbl} --adaptive_margin --tta --out_csv '{cross_csv}'",
                desc="Zero-Shot Cross-Dataset Transfer Evaluation (trained on BraTS2020, tested on secondary cohort)")
        summary_records["9. Cross-Dataset Transfer (A)"] = summarize_csv(cross_csv)

    # Compile Summary Tables
    md_summary = dict_to_markdown_table(summary_records)

    print("\n" + "="*80)
    print("ADVANCED EVALUATION SUMMARY (A, B, C):")
    print("="*80)
    print(md_summary)

    out_md = os.path.join(results_dir, "ADVANCED_RESULTS.md")
    with open(out_md, "w") as f:
        f.write("# Advanced Results: Generalization (A), Adaptive Margin (B), and TTA (C)\n\n" + md_summary + "\n")

    # Save checkpoints into bundle so weights are preserved
    bundle_ckpt_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(bundle_ckpt_dir, exist_ok=True)
    shutil.copy2(s1_path, os.path.join(bundle_ckpt_dir, "s1_best.pt"))
    shutil.copy2(s2_path, os.path.join(bundle_ckpt_dir, "s2_best.pt"))

    # Bundle all advanced results into zip
    zip_path = os.path.join(work_dir, "advanced_results_bundle.zip")
    shutil.make_archive(zip_path.replace(".zip", ""), 'zip', results_dir)
    print(f"\nAll advanced results, tables, and model checkpoints bundled to: {zip_path}")

if __name__ == "__main__":
    main()
