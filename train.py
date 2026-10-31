"""Train either stage.

  python train.py --stage 1 --cache cache --out runs/s1 --epochs 150
  python train.py --stage 2 --cache cache --out runs/s2 --epochs 300 --skip_attn cbam
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import CoarseDataset, FineDataset, list_cases, modality_dropout, split_cases
from losses import DiceBoundaryLoss, soft_dice
from models import CoarseNet, FineNet, pad_to_multiple


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--base", type=int, default=None, help="base channels (s1 default 8, s2 default 16)")
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--skip_attn", default="cbam", choices=["cbam", "gate", "none"])
    ap.add_argument("--tf_layers", type=int, default=2)
    ap.add_argument("--moddrop", type=float, default=0.3, help="prob. of modality dropout per sample")
    ap.add_argument("--et_label", type=int, default=4, help="4 for BraTS<=2021, 3 for BraTS2023")
    ap.add_argument("--no_dilated", action="store_true", help="disable dilated convs (for ablation A4)")
    ap.add_argument("--w_bd", type=float, default=0.5, help="weight of boundary loss (0 for ablation A6)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def build(args):
    if args.stage == 1:
        return CoarseNet(base=args.base or 8)
    dilated_stages = () if getattr(args, "no_dilated", False) else (3, 4)
    return FineNet(base=args.base or 16, skip_attn=args.skip_attn,
                   dilated_stages=dilated_stages, tf_layers=args.tf_layers)


def forward_loss(model, x, y, crit, stage):
    if stage == 1:
        x, _ = pad_to_multiple(x, 8)
        y, _ = pad_to_multiple(y, 8)
        return crit(model(x), y), None
    x, _ = pad_to_multiple(x, 16)
    y, _ = pad_to_multiple(y, 16)
    out, aux = model(x, return_aux=True)
    y_aux = F.interpolate(y, size=aux.shape[2:], mode="nearest")
    return crit(out, y) + 0.4 * crit(aux, y_aux), out


@torch.no_grad()
def validate(model, loader, dev, stage):
    model.eval()
    dices = []
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        m = 8 if stage == 1 else 16
        xp, _ = pad_to_multiple(x, m)
        out = model(xp)[..., :x.shape[2], :x.shape[3], :x.shape[4]]
        pred = (torch.sigmoid(out) > 0.5).float()
        dices.append(soft_dice(pred, y))
    d = torch.stack(dices).mean(0).cpu()
    return d  # (C,)


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    train_c, val_c, test_c = split_cases(list_cases(args.cache), seed=args.seed)
    json.dump({"train": train_c, "val": val_c, "test": test_c}, open(os.path.join(args.out, "split.json"), "w"))
    print(f"cases: train {len(train_c)}  val {len(val_c)}  test {len(test_c)}")

    if args.stage == 1:
        ds_tr = CoarseDataset(args.cache, train_c, train=True, et_label=args.et_label)
        ds_va = CoarseDataset(args.cache, val_c, train=False, et_label=args.et_label)
    else:
        ds_tr = FineDataset(args.cache, train_c, patch=args.patch, train=True, et_label=args.et_label)
        ds_va = FineDataset(args.cache, val_c, patch=args.patch, train=False, et_label=args.et_label)
    tr = DataLoader(ds_tr, args.bs, shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True)
    va = DataLoader(ds_va, 1, shuffle=False, num_workers=args.workers)

    model = build(args).to(dev)
    print("params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
    crit = DiceBoundaryLoss(w_bd=args.w_bd)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(tr),
                                                pct_start=0.05)
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = -1.0
    for ep in range(args.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for x, y in tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            if args.moddrop > 0:
                x = modality_dropout(x, args.moddrop)
            with torch.autocast(dev, dtype=torch.float16, enabled=use_amp):
                loss, _ = forward_loss(model, x, y, crit, args.stage)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
        d = validate(model, va, dev, args.stage)
        score = d.mean().item()
        names = ["WT"] if args.stage == 1 else ["WT", "TC", "ET"]
        print(f"ep {ep+1:3d}  loss {run/len(tr):.4f}  val " +
              " ".join(f"{n} {v:.4f}" for n, v in zip(names, d.tolist())) + f"  ({time.time()-t0:.0f}s)")
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "args": vars(args), "val": d.tolist()},
                       os.path.join(args.out, "best.pt"))


if __name__ == "__main__":
    main()
