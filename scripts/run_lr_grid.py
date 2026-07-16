"""Chapter 3 hyperparameter selection: small lr/weight-decay grid on validation.

Run once, on seed 42 with augmentation, for each architecture; the chosen
values are then frozen into the 12-run experiment configs. Never touches the
test set.

Usage: python scripts/run_lr_grid.py
"""
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train import run_training  # noqa: E402

LRS = [3e-4, 1e-3]
WDS = [0.01, 0.05]


def main() -> None:
    results = []
    for arch in ("transformer", "lstm"):
        base = yaml.safe_load(
            Path(f"configs/{arch}_aug_s42.yaml").read_text(encoding="utf-8")
        )
        for lr in LRS:
            for wd in WDS:
                cfg = dict(base)
                cfg["lr"] = lr
                cfg["weight_decay"] = wd
                cfg["run_id"] = f"{arch}_lr{lr:g}_wd{wd:g}"
                cfg["out_dir"] = "results/lr_grid"
                out = run_training(cfg)
                results.append((arch, lr, wd, out["best_val_acc"]))
                print(
                    f"[lr-grid] {arch} lr={lr:g} wd={wd:g} "
                    f"-> best val acc {out['best_val_acc']:.4f}",
                    flush=True,
                )

    print("\n=== lr/wd selection summary ===")
    for arch in ("transformer", "lstm"):
        rows = [r for r in results if r[0] == arch]
        rows.sort(key=lambda r: -r[3])
        for arch_, lr, wd, acc in rows:
            print(f"{arch_:<12} lr={lr:<8g} wd={wd:<6g} val acc {acc:.4f}")
        best = rows[0]
        print(f"BEST {arch}: lr={best[1]:g} wd={best[2]:g} ({best[3]:.4f})\n")


if __name__ == "__main__":
    main()
