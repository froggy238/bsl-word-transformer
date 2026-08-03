"""Build the v2 data partition: organisation-held-out TEST set, clip-level val.

Context (2026-07-27): the methodology's self-recorded test
set was replaced by promoting the v1 validation organisations — an
organisation-grouped hold-out no model was ever trained on — to the one-shot
test partition. Model selection moves to a new clip-level validation split cut
inside the 11 training organisations, and the 12-run grid is re-trained against
it (configs_v2/ -> results/runs_v2/) so no finally-selected model was ever
chosen using test clips.

Reads   data/splits.json          (v1: organisation-grouped train/val, frozen)
Writes  data/splits_v2.json       train/val over the former train pool;
                                  the former val (69 clips, 3 orgs) recorded as test
        data/test_landmarks/      copies of the test clips' cached landmarks,
                                  one folder per word (evaluate.py --split test layout)
        configs_v2/*.yaml         the 12-run grid re-pointed at splits_v2.json
                                  and results/runs_v2

The new validation split takes exactly one clip per word (sampled with
numpy default_rng(SEED) over the per-word sorted clip lists), so every class is
validated while at most one clip per class leaves training. Unlike v1 this val
shares organisations with train — it is a model-selection split only; the
organisation-disjoint generalisation measurement lives in the test partition.

Usage:
    python -m src.make_v2_partition
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from src.dataset import clip_id_to_word

SEED = 42
SPLITS_V1 = Path("data/splits.json")
SPLITS_V2 = Path("data/splits_v2.json")
LANDMARKS_DIR = Path("data/landmarks")
TEST_LANDMARKS_DIR = Path("data/test_landmarks")
CONFIGS_V1 = Path("configs")
CONFIGS_V2 = Path("configs_v2")


def main() -> None:
    with open(SPLITS_V1, encoding="utf-8") as f:
        v1 = json.load(f)
    label_list: list[str] = v1["label_list"]
    pool: list[str] = list(v1["train"])  # 11 training organisations
    test: list[str] = list(v1["val"])  # 3 held-out organisations -> test

    # Group the pool by word; one seeded draw per word forms the new val.
    by_word: dict[str, list[str]] = {}
    for clip_id in pool:
        by_word.setdefault(clip_id_to_word(clip_id), []).append(clip_id)
    missing = [w for w in label_list if w not in by_word]
    assert not missing, f"words with no training clips: {missing}"

    rng = np.random.default_rng(SEED)
    val: list[str] = []
    # Iterate label_list (fixed order) and sort each word's clips so the draw
    # depends only on SEED, never on JSON ordering.
    for word in label_list:
        clips = sorted(by_word[word])
        val.append(clips[int(rng.integers(len(clips)))])
    val_set = set(val)
    train = [c for c in pool if c not in val_set]

    # Invariants: disjoint partitions, nothing lost, full train coverage.
    assert not (set(train) & val_set) and not (set(pool) & set(test))
    assert len(train) + len(val) == len(pool)
    train_words = {clip_id_to_word(c) for c in train}
    assert all(w in train_words for w in label_list)
    test_words = {clip_id_to_word(c) for c in test}
    untested = [w for w in label_list if w not in test_words]

    v2 = {
        "seed": SEED,
        "group_by": "clip_stratified_within_train_orgs",
        "derived_from": str(SPLITS_V1),
        "test_orgs": v1["val_orgs"],  # promoted organisation hold-out
        "label_list": label_list,
        "train": train,
        "val": val,
        "test": test,
    }
    with open(SPLITS_V2, "w", encoding="utf-8") as f:
        json.dump(v2, f, indent=2)

    # Materialise the test partition in the folder-per-word layout that
    # `python -m src.evaluate --split test` discovers (copy, never move).
    copied = 0
    for clip_id in test:
        word = clip_id_to_word(clip_id)
        src = LANDMARKS_DIR / word / f"{clip_id}.npz"
        assert src.is_file(), f"missing landmark cache: {src}"
        dst_dir = TEST_LANDMARKS_DIR / word
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{clip_id}.npz"
        if not dst.is_file():
            shutil.copy2(src, dst)
        copied += 1

    # Re-point the 12-run grid; everything else in each config stays identical.
    CONFIGS_V2.mkdir(exist_ok=True)
    n_cfg = 0
    for cfg_path in sorted(CONFIGS_V1.glob("*.yaml")):
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["splits_file"] = str(SPLITS_V2).replace("\\", "/")
        cfg["out_dir"] = "results/runs_v2"
        with open(CONFIGS_V2 / cfg_path.name, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        n_cfg += 1

    print(
        f"splits_v2: {len(train)} train / {len(val)} val / {len(test)} test clips"
    )
    print(f"test partition: {copied} landmark files under {TEST_LANDMARKS_DIR}")
    print(f"configs: {n_cfg} written to {CONFIGS_V2}")
    if untested:
        print(
            f"NOTE: {len(untested)} class(es) have zero test clips "
            f"(known v1 gap): {', '.join(untested)}"
        )


if __name__ == "__main__":
    main()
