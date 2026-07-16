"""Generate the 12 experiment configs: {arch} x {aug,noaug} x seeds {42,43,44}.

Run: python configs/gen_configs.py  (writes the YAML files next to this script)
"""

from pathlib import Path

TEMPLATE = """\
run_id: {run_id}
arch: {arch}
augment: {augment}
seed: {seed}
epochs: 80
batch_size: 32
lr: 3.0e-4
weight_decay: 0.01
warmup_epochs: 5
label_smoothing: 0.1
seq_len: 64
in_dim: 315
n_classes: 50
d_model: 192
n_layers: 4
n_heads: 6
d_ff: 384
dropout: 0.1
lstm_hidden: 256
lstm_layers: 2
landmarks_dir: data/landmarks
splits_file: data/splits.json
metadata_csv: data/metadata.csv
out_dir: results/runs
"""

ARCHES = ("transformer", "lstm")
SEEDS = (42, 43, 44)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    for arch in ARCHES:
        for augment in (True, False):
            for seed in SEEDS:
                tag = "aug" if augment else "noaug"
                run_id = f"{arch}_{tag}_s{seed}"
                path = out_dir / f"{run_id}.yaml"
                path.write_text(
                    TEMPLATE.format(
                        run_id=run_id,
                        arch=arch,
                        augment="true" if augment else "false",
                        seed=seed,
                    ),
                    encoding="utf-8",
                )
                print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
