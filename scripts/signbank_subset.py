"""SignBank cross-corpus test: overall vs high-confidence-subset accuracy.

Run from the repo root AFTER the one-shot SignBank evaluation:
    python scripts/signbank_subset.py

Joins each selected run's eval_test_signbank/predictions.csv with the download
provenance (data/signbank_metadata.csv) and the curated gloss map to report
top-1/top-5 on (a) all clips, (b) the exact-keyword-match (high-confidence)
subset, (c) excluding the four sense-flagged words where SignBank has no
same-gloss entry (stop=FINISH, today=NOW, need=WANT, goodbye=WAVE-HAND).
"""
from pathlib import Path

import pandas as pd

RUNS = ["transformer_noaug_s42", "transformer_aug_s42", "lstm_aug_s42", "lstm_noaug_s42"]
FLAGGED_WORDS = {"stop", "today", "need", "goodbye"}

prov = pd.read_csv("data/signbank_metadata.csv")
cur = pd.read_csv("data/signbank_gloss_map_curated.csv")
prov["clip_id"] = prov["filename"].str.replace(".mp4", "", regex=False)
conf = prov.merge(cur[["video_url", "match_confidence"]], on="video_url", how="left")
conf = conf[["clip_id", "match_confidence", "gloss"]].drop_duplicates("clip_id")

for run in RUNS:
    preds_path = Path("results/runs_v2") / run / "eval_test_signbank" / "predictions.csv"
    if not preds_path.is_file():
        print(f"{run}: no eval_test_signbank/predictions.csv - skipped")
        continue
    df = pd.read_csv(preds_path).merge(conf, on="clip_id", how="left")
    hi = df[df["match_confidence"] == "high"]
    top5_cols = [f"top5_{i}" for i in range(1, 6)]
    top5 = df.apply(lambda r: r["true"] in [r[c] for c in top5_cols], axis=1)
    top5_hi = hi.apply(lambda r: r["true"] in [r[c] for c in top5_cols], axis=1)
    unflagged = df[~df["true"].isin(FLAGGED_WORDS)]
    print(f"{run}:")
    print(f"  all rows      n={len(df):3d}  top1={df['correct'].mean():.4f}  top5={top5.mean():.4f}")
    print(f"  high-conf     n={len(hi):3d}  top1={hi['correct'].mean():.4f}  top5={top5_hi.mean():.4f}")
    print(f"  excl flagged  n={len(unflagged):3d}  top1={unflagged['correct'].mean():.4f}")
