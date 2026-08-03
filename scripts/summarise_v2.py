"""Summarise the v2 grid: per-run best val acc, group stats, best run per group.

Run from the repo root: python scripts/summarise_v2.py
Writes results/validation_summary_v2.csv and prints the per-group selection.
"""
import csv
from pathlib import Path

import numpy as np
import torch

RUNS = Path("results/runs_v2")
OUT = Path("results/validation_summary_v2.csv")

rows = []
for run_dir in sorted(p for p in RUNS.iterdir() if p.is_dir()):
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.is_file() or run_dir.name.endswith("_shufflelabels"):
        continue
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    rows.append({
        "run_id": run_dir.name,
        "arch": cfg["arch"],
        "augment": bool(cfg.get("augment")),
        "seed": int(cfg["seed"]),
        "best_val_acc": float(ckpt["val_acc"]),
        "best_epoch": int(ckpt["epoch"]),
    })

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {OUT} ({len(rows)} runs)\n")

groups: dict[tuple, list[dict]] = {}
for r in rows:
    groups.setdefault((r["arch"], r["augment"]), []).append(r)
print(f"{'group':<24} {'mean':>6} {'std':>6}  best run (val acc, epoch)")
for (arch, aug), rs in sorted(groups.items()):
    accs = [r["best_val_acc"] for r in rs]
    best = max(rs, key=lambda r: (r["best_val_acc"], -r["seed"]))
    print(
        f"{arch + ('_aug' if aug else '_noaug'):<24} "
        f"{np.mean(accs):>6.3f} {np.std(accs):>6.3f}  "
        f"{best['run_id']} ({best['best_val_acc']:.3f}, ep {best['best_epoch']})"
    )
