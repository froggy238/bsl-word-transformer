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

# 2x2 grid around common AdamW settings; deliberately small because every cell is a full
# training run, and Chapter 3 needs a defensible choice rather than an exhaustive sweep
LRS = [3e-4, 1e-3]
WDS = [0.01, 0.05]


def main() -> None:
    results = []
    # tune each architecture separately: a fair comparison needs each model at its own best
    # lr/wd, not the transformer's settings imposed on the parameter-matched LSTM
    for arch in ("transformer", "lstm"):
        # template = seed 42 + augmentation: hyperparameters are chosen once under the
        # intended training regime (aug on, one representative seed); sweeping every
        # seed/aug combo would multiply compute and risk tuning to seed noise
        base = yaml.safe_load(
            Path(f"configs/{arch}_aug_s42.yaml").read_text(encoding="utf-8")
        )
        for lr in LRS:
            for wd in WDS:
                cfg = dict(base)  # shallow copy so grid cells never mutate the template
                cfg["lr"] = lr
                cfg["weight_decay"] = wd
                cfg["run_id"] = f"{arch}_lr{lr:g}_wd{wd:g}"  # :g gives compact ids (lr0.001)
                cfg["out_dir"] = "results/lr_grid"  # quarantine sweep outputs from main results
                # selection metric is best VALIDATION accuracy on the org-grouped split; the
                # held-out test set is never loaded here, preserving the evaluate-once rule
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
        rows.sort(key=lambda r: -r[3])  # r[3] = best val acc; descending, best first
        for arch_, lr, wd, acc in rows:
            print(f"{arch_:<12} lr={lr:<8g} wd={wd:<6g} val acc {acc:.4f}")
        best = rows[0]
        print(f"BEST {arch}: lr={best[1]:g} wd={best[2]:g} ({best[3]:.4f})\n")


if __name__ == "__main__":
    main()
