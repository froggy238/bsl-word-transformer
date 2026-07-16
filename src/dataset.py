"""Dataset layer: clip loading, temporal resampling, PyTorch dataset, splits.

Pipeline per clip: load cached landmarks (.npz) -> fill short NaN gaps ->
per-frame normalisation -> NaN->0 -> (optional temporal augmentation) ->
fixed-length resampling to 64 frames -> (optional spatial augmentation) ->
flatten to (64, 315) features.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.augment import spatial_augment, temporal_augment
from src.extract import fill_gaps
from src.landmarks import FEATURES_PER_FRAME
from src.normalise import normalise_sequence

DEFAULT_SPLITS_PATH = "data/splits.json"


def resample_sequence(seq: np.ndarray, target_len: int = 64) -> np.ndarray:
    """Linearly resample a sequence along time to a fixed length.

    Args:
        seq: Array of shape (T, 105, 3) (any finite values; not NaN-safe).
        target_len: Output length.

    Returns:
        float32 array of shape (target_len, 105, 3). Endpoints are
        preserved exactly (frame 0 and frame T-1 map to output ends).
    """
    seq = np.asarray(seq)
    if seq.ndim != 3:
        raise ValueError(f"expected 3-d array (T, L, C), got shape {seq.shape}")
    t = seq.shape[0]
    if t == 0:
        raise ValueError("cannot resample an empty sequence")
    if t == 1:
        return np.repeat(seq, target_len, axis=0).astype(np.float32)

    pos = np.linspace(0.0, t - 1, num=target_len)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, t - 1)
    w = (pos - lo).astype(np.float32)[:, None, None]
    out = (1.0 - w) * seq[lo] + w * seq[hi]
    return out.astype(np.float32)


def load_clip(npz_path: str | Path) -> np.ndarray:
    """Load a cached landmark clip and apply gap filling + normalisation.

    Args:
        npz_path: Path to a {word}/{clip_id}.npz cache file with keys
            'landmarks' (T, 105, 3), 'presence' (T, 3) and 'fps'.

    Returns:
        float32 array of shape (T, 105, 3): gaps of <= 5 frames linearly
        interpolated, sequence normalised per frame, remaining NaN set to 0.
    """
    with np.load(npz_path) as data:
        landmarks = data["landmarks"].astype(np.float32)
        presence = data["presence"].astype(np.float32)
    seq = fill_gaps(landmarks, presence)
    seq = normalise_sequence(seq)
    return np.nan_to_num(seq, nan=0.0).astype(np.float32)


def clip_id_to_word(clip_id: str) -> str:
    """Recover the label word from a clip id of the form {word}_{source}_{nnn}."""
    return clip_id.rsplit("_", 2)[0]


class BSLDataset(Dataset):
    """Word-level BSL clip dataset over cached landmark files.

    Labels are passed in explicitly so the dataset can be constructed from
    splits.json + label_list alone, without metadata. Uses a single numpy
    Generator for augmentation randomness; the project runs all DataLoaders
    with num_workers=0, so no per-worker reseeding is needed.
    """

    def __init__(
        self,
        clip_ids: list[str],
        labels: list[int],
        landmarks_dir: str,
        metadata: pd.DataFrame | None = None,
        augment: bool = False,
        seq_len: int = 64,
        seed: int = 42,
    ) -> None:
        if len(clip_ids) != len(labels):
            raise ValueError(
                f"clip_ids ({len(clip_ids)}) and labels ({len(labels)}) "
                "must have the same length"
            )
        self.clip_ids = list(clip_ids)
        self.labels = [int(l) for l in labels]
        self.landmarks_dir = Path(landmarks_dir)
        self.metadata = metadata
        self.augment = augment
        self.seq_len = seq_len
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.clip_ids)

    def npz_path(self, clip_id: str) -> Path:
        """Path of the cached landmark file for a clip id."""
        return self.landmarks_dir / clip_id_to_word(clip_id) / f"{clip_id}.npz"

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        clip_id = self.clip_ids[index]
        seq = load_clip(self.npz_path(clip_id))
        if self.augment:
            seq = temporal_augment(seq, self.rng)
        seq = resample_sequence(seq, self.seq_len)
        if self.augment:
            seq = spatial_augment(seq, self.rng)
        features = np.ascontiguousarray(
            seq.reshape(self.seq_len, FEATURES_PER_FRAME), dtype=np.float32
        )
        return torch.from_numpy(features), self.labels[index]


def _grouped_split(
    metadata: pd.DataFrame,
    label_list: list[str],
    val_frac: float,
    seed: int,
) -> dict:
    """One candidate organisation-grouped split for a given shuffle seed."""
    orgs = sorted(metadata["organisation"].astype(str).unique().tolist())
    if len(orgs) < 2:
        raise ValueError("need at least 2 organisations for a grouped split")
    rng = np.random.default_rng(seed)
    rng.shuffle(orgs)

    org_counts = metadata.groupby(metadata["organisation"].astype(str))[
        "clip_id"
    ].count()
    total = int(len(metadata))
    target = val_frac * total

    # Size-aware greedy fill: an organisation joins val only if that brings
    # the val clip count strictly closer to the target, so one large source
    # drawn early cannot blow the split past ~val_frac.
    val_orgs: list[str] = []
    val_count = 0
    for org in orgs:
        count = int(org_counts[org])
        if abs(val_count + count - target) < abs(val_count - target):
            val_orgs.append(org)
            val_count += count
    if not val_orgs:
        # Every organisation overshoots the target; take the smallest.
        smallest = min(orgs, key=lambda o: (int(org_counts[o]), o))
        val_orgs.append(smallest)
        val_count = int(org_counts[smallest])
    if len(val_orgs) == len(orgs):
        # Never leave the train side empty; smallest val_org goes back.
        moved = min(val_orgs, key=lambda o: (int(org_counts[o]), o))
        val_orgs.remove(moved)
        val_count -= int(org_counts[moved])

    is_val = metadata["organisation"].astype(str).isin(set(val_orgs))
    words = metadata["word"].astype(str)
    train_words = set(words[~is_val])
    val_words = set(words[is_val])
    return {
        "seed": seed,
        "val_orgs": val_orgs,
        "train": metadata.loc[~is_val, "clip_id"].astype(str).tolist(),
        "val": metadata.loc[is_val, "clip_id"].astype(str).tolist(),
        "achieved": val_count / total if total else 0.0,
        "missing_train": [w for w in label_list if w not in train_words],
        "missing_val": [w for w in label_list if w not in val_words],
    }


def make_splits(
    metadata_csv: str,
    vocabulary_csv: str,
    val_frac: float = 0.2,
    seed: int = 42,
    search_seeds: int = 0,
    val_orgs: list[str] | None = None,
) -> dict:
    """Build a grouped train/val split keyed by organisation.

    Organisations are shuffled deterministically (seed) and assigned
    greedily to the validation side, aiming at ``val_frac`` of clips.
    No organisation appears on both sides.

    With ``search_seeds`` > 0, shuffle seeds 0..search_seeds-1 are tried and
    the best split kept, ranked by: every class present in train (mandatory
    for training to see all 50 signs), fewest classes absent from val (the
    handbook requires every class in validation where the grouping allows
    it), then closest achieved fraction to ``val_frac``. The chosen seed is
    recorded in the output. Warns about any residual coverage gaps.

    With ``val_orgs``, the validation organisations are set explicitly
    (chosen offline, e.g. by exhaustive subset enumeration) and no shuffle
    happens; the choice is recorded verbatim in the output for audit.

    Returns:
        Dict with keys seed, group_by, val_orgs, label_list, train, val
        (the data/splits.json schema).
    """
    metadata = pd.read_csv(metadata_csv)
    vocabulary = pd.read_csv(vocabulary_csv)
    label_list = sorted(vocabulary["word"].astype(str).unique().tolist())

    if val_orgs is not None:
        known = set(metadata["organisation"].astype(str))
        unknown = sorted(set(val_orgs) - known)
        if unknown:
            raise ValueError(f"unknown organisation(s): {', '.join(unknown)}")
        is_val = metadata["organisation"].astype(str).isin(set(val_orgs))
        words = metadata["word"].astype(str)
        train_words, val_words = set(words[~is_val]), set(words[is_val])
        best = {
            "seed": seed,
            "val_orgs": sorted(val_orgs),
            "train": metadata.loc[~is_val, "clip_id"].astype(str).tolist(),
            "val": metadata.loc[is_val, "clip_id"].astype(str).tolist(),
            "achieved": float(is_val.mean()),
            "missing_train": [w for w in label_list if w not in train_words],
            "missing_val": [w for w in label_list if w not in val_words],
        }
    else:
        candidate_seeds = (
            list(range(search_seeds)) if search_seeds > 0 else [seed]
        )
        best = None
        best_key: tuple | None = None
        for s in candidate_seeds:
            result = _grouped_split(metadata, label_list, val_frac, s)
            key = (
                len(result["missing_train"]),
                len(result["missing_val"]),
                abs(result["achieved"] - val_frac),
                s,
            )
            if best_key is None or key < best_key:
                best, best_key = result, key
    assert best is not None

    if val_orgs is None and search_seeds > 0:
        print(
            f"seed search over {search_seeds} shuffles: chose seed "
            f"{best['seed']} (val orgs: {', '.join(best['val_orgs'])})"
        )
    if best["missing_train"]:
        msg = (
            f"{len(best['missing_train'])} class(es) have ZERO training "
            f"clips: {', '.join(best['missing_train'])}"
        )
        print(f"WARNING: {msg}")
        warnings.warn(msg)
    if best["missing_val"]:
        msg = (
            f"{len(best['missing_val'])} class(es) absent from val split: "
            f"{', '.join(best['missing_val'])}"
        )
        print(f"WARNING: {msg}")
        warnings.warn(msg)
    if abs(best["achieved"] - val_frac) > 0.10:
        warnings.warn(
            f"achieved val fraction {best['achieved']:.2f} deviates from "
            f"target {val_frac:.2f} by more than 0.10; organisation sizes "
            "are skewed"
        )

    return {
        "seed": best["seed"],
        "group_by": "organisation",
        "val_orgs": best["val_orgs"],
        "label_list": label_list,
        "train": best["train"],
        "val": best["val"],
    }


def load_splits(path: str = DEFAULT_SPLITS_PATH) -> dict:
    """Load a splits dict written by --make-splits."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _print_summary(metadata_csv: str) -> None:
    metadata = pd.read_csv(metadata_csv)
    counts = metadata.groupby(metadata["word"].astype(str))["clip_id"].count()
    print(f"{'word':<20} clips")
    for word, count in counts.sort_index().items():
        print(f"{word:<20} {count}")
    print(f"{'TOTAL':<20} {len(metadata)} clips, {counts.size} classes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset utilities.")
    parser.add_argument(
        "--make-splits",
        action="store_true",
        help="build the grouped train/val split and write it to --out",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print per-class clip counts from the metadata CSV",
    )
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--vocabulary", default="data/vocabulary.csv")
    parser.add_argument("--out", default=DEFAULT_SPLITS_PATH)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--search-seeds",
        type=int,
        default=0,
        help="try this many shuffle seeds and keep the best-covered split",
    )
    parser.add_argument(
        "--val-orgs",
        default=None,
        help="comma-separated organisations to place in val (explicit split)",
    )
    args = parser.parse_args()

    if not (args.make_splits or args.summary):
        parser.error("nothing to do: pass --make-splits and/or --summary")

    if args.summary:
        _print_summary(args.metadata)

    if args.make_splits:
        splits = make_splits(
            args.metadata,
            args.vocabulary,
            val_frac=args.val_frac,
            seed=args.seed,
            search_seeds=args.search_seeds,
            val_orgs=(
                [o.strip() for o in args.val_orgs.split(",") if o.strip()]
                if args.val_orgs
                else None
            ),
        )
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2)
        n_train, n_val = len(splits["train"]), len(splits["val"])
        frac = n_val / max(n_train + n_val, 1)
        print(
            f"wrote {out_path}: {n_train} train / {n_val} val clips "
            f"(val fraction {frac:.2f}), {len(splits['label_list'])} classes"
        )


if __name__ == "__main__":
    main()
