#!/usr/bin/env bash
# Runs the label-shuffle sanity check, then the full 12-run experiment grid.
# Usage: bash scripts/run_grid.sh [python-executable]
set -u
PY="${1:-python}"
cd "$(dirname "$0")/.."

echo "=== Label-shuffle sanity check (val acc should sit near 2%) ==="
"$PY" -m src.train --config configs/transformer_aug_s42.yaml \
  --shuffle-labels --epochs 20 --device cpu

echo
echo "=== 12-run grid ==="
for cfg in configs/transformer_aug_s42.yaml configs/transformer_aug_s43.yaml \
           configs/transformer_aug_s44.yaml configs/transformer_noaug_s42.yaml \
           configs/transformer_noaug_s43.yaml configs/transformer_noaug_s44.yaml \
           configs/lstm_aug_s42.yaml configs/lstm_aug_s43.yaml \
           configs/lstm_aug_s44.yaml configs/lstm_noaug_s42.yaml \
           configs/lstm_noaug_s43.yaml configs/lstm_noaug_s44.yaml; do
  echo "--- $cfg ---"
  "$PY" -m src.train --config "$cfg" --device cpu || echo "FAILED: $cfg"
done

echo
echo "=== Grid summary (best val acc per run) ==="
"$PY" - <<'EOF'
import json
from pathlib import Path

import torch

for run_dir in sorted(Path("results/runs").iterdir()):
    ckpt = run_dir / "best.pt"
    if ckpt.is_file():
        c = torch.load(ckpt, map_location="cpu")
        print(f"{run_dir.name:<28} best val acc {c['val_acc']:.4f} "
              f"(epoch {c['epoch']})")
EOF
