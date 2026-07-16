"""Tests for src.dataset using synthetic landmark arrays only (no real data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.dataset import BSLDataset, load_clip, make_splits, resample_sequence
from src.landmarks import (
    LEFT_HAND_SLICE,
    LEFT_SHOULDER,
    N_COORDS,
    N_LANDMARKS,
    RIGHT_HAND_SLICE,
    RIGHT_SHOULDER,
)

METADATA_COLUMNS = [
    "word", "clip_id", "source", "organisation", "signer_id", "source_url",
    "video_file", "resolution", "duration_s", "fps", "download_date", "notes",
]


def _synthetic_landmarks(t: int, seed: int = 0) -> np.ndarray:
    """Random (t, 105, 3) landmarks with constant, finite shoulders."""
    rng = np.random.default_rng(seed)
    seq = rng.uniform(0.2, 0.8, size=(t, N_LANDMARKS, N_COORDS)).astype(np.float32)
    seq[:, LEFT_SHOULDER] = [0.4, 0.5, 0.0]
    seq[:, RIGHT_SHOULDER] = [0.6, 0.5, 0.0]
    return seq


def _write_npz(
    path: Path, landmarks: np.ndarray, presence: np.ndarray, fps: float = 25.0
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        landmarks=landmarks.astype(np.float32),
        presence=presence.astype(np.float32),
        fps=np.float32(fps),
    )


# ---------------------------------------------------------------------------
# (a) resample_sequence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("t", [5, 64, 200])
def test_resample_sequence_shape_and_endpoints(t: int) -> None:
    offsets = np.random.default_rng(1).normal(size=(N_LANDMARKS, N_COORDS))
    ramp = np.linspace(0.0, 1.0, t, dtype=np.float64)[:, None, None]
    seq = (ramp + offsets[None]).astype(np.float32)

    out = resample_sequence(seq, target_len=64)

    assert out.shape == (64, N_LANDMARKS, N_COORDS)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[0], seq[0], atol=1e-5)
    np.testing.assert_allclose(out[-1], seq[-1], atol=1e-5)
    # Linear interpolation reproduces a linear-in-time signal exactly.
    expected = np.linspace(0.0, 1.0, 64)[:, None, None] + offsets[None]
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_resample_sequence_identity_length() -> None:
    seq = _synthetic_landmarks(64)
    out = resample_sequence(seq, target_len=64)
    np.testing.assert_allclose(out, seq, atol=1e-6)


# ---------------------------------------------------------------------------
# (b) load_clip
# ---------------------------------------------------------------------------


def test_load_clip_fills_short_gap_and_zeroes_long_gap(tmp_path: Path) -> None:
    t = 30
    landmarks = _synthetic_landmarks(t, seed=2)
    presence = np.ones((t, 3), dtype=np.float32)

    # 3-frame left-hand gap (frames 10-12): should be linearly interpolated.
    landmarks[10:13, LEFT_HAND_SLICE] = np.nan
    presence[10:13, 0] = 0.0
    # 8-frame right-hand gap (frames 18-25): > 5, stays NaN then becomes 0.
    landmarks[18:26, RIGHT_HAND_SLICE] = np.nan
    presence[18:26, 1] = 0.0

    path = tmp_path / "hello" / "hello_signbsl_001.npz"
    _write_npz(path, landmarks, presence)

    out = load_clip(path)

    assert out.shape == (t, N_LANDMARKS, N_COORDS)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()

    # Shoulders are constant, so per-frame normalisation is the same affine
    # map in every frame and a linear fill in raw space stays linear in
    # normalised space: check frames 10-12 against interp of frames 9 and 13.
    a, b = out[9, LEFT_HAND_SLICE], out[13, LEFT_HAND_SLICE]
    for k, frame in enumerate(range(10, 13), start=1):
        w = k / 4.0
        np.testing.assert_allclose(
            out[frame, LEFT_HAND_SLICE], (1 - w) * a + w * b, atol=1e-5
        )
    # Filled values must not be zero (they are genuine interpolated points).
    assert np.abs(out[10:13, LEFT_HAND_SLICE]).max() > 0

    # The long gap is zeroed after normalisation.
    assert np.all(out[18:26, RIGHT_HAND_SLICE] == 0.0)
    # Frames outside the gap keep real (non-zero) right-hand values.
    assert np.abs(out[17, RIGHT_HAND_SLICE]).max() > 0
    assert np.abs(out[26, RIGHT_HAND_SLICE]).max() > 0


# ---------------------------------------------------------------------------
# (c) BSLDataset
# ---------------------------------------------------------------------------


def test_bsl_dataset_shapes_and_determinism(tmp_path: Path) -> None:
    landmarks_dir = tmp_path / "landmarks"
    clips = {"hello_signbsl_001": ("hello", 25), "thank-you_signbsl_001": ("thank-you", 40)}
    for i, (clip_id, (word, t)) in enumerate(clips.items()):
        _write_npz(
            landmarks_dir / word / f"{clip_id}.npz",
            _synthetic_landmarks(t, seed=10 + i),
            np.ones((t, 3), dtype=np.float32),
        )

    ds = BSLDataset(
        clip_ids=list(clips.keys()),
        labels=[0, 1],
        landmarks_dir=str(landmarks_dir),
        metadata=None,
        augment=False,
        seq_len=64,
        seed=42,
    )

    assert len(ds) == 2
    features, label = ds[0]
    assert isinstance(features, torch.Tensor)
    assert features.shape == (64, 315)
    assert features.dtype == torch.float32
    assert isinstance(label, int)
    assert label == 0
    _, label1 = ds[1]
    assert label1 == 1

    epoch_a = [ds[i] for i in range(len(ds))]
    epoch_b = [ds[i] for i in range(len(ds))]
    for (xa, ya), (xb, yb) in zip(epoch_a, epoch_b):
        assert torch.equal(xa, xb)
        assert ya == yb


# ---------------------------------------------------------------------------
# (d) make_splits
# ---------------------------------------------------------------------------


def _synthetic_metadata(tmp_path: Path) -> tuple[str, str, pd.DataFrame]:
    words = ["hello", "thank-you", "please", "sorry", "help"]
    orgs = ["org-a", "org-b", "org-c", "org-d"]
    rows = []
    for word in words:
        for oi, org in enumerate(orgs):
            for k in range(2):
                clip_id = f"{word}_src{oi}_{k:03d}"
                rows.append({
                    "word": word,
                    "clip_id": clip_id,
                    "source": f"src{oi}",
                    "organisation": org,
                    "signer_id": f"signer{oi}",
                    "source_url": f"https://example.org/{clip_id}",
                    "video_file": f"data/raw_videos/{word}/{clip_id}.mp4",
                    "resolution": "640x480",
                    "duration_s": 2.5,
                    "fps": 25.0,
                    "download_date": "2026-07-01",
                    "notes": "",
                })
    metadata = pd.DataFrame(rows, columns=METADATA_COLUMNS)
    metadata_csv = tmp_path / "metadata.csv"
    metadata.to_csv(metadata_csv, index=False)

    vocabulary = pd.DataFrame(
        {"word": words, "category": "core", "handedness": "one", "notes": ""}
    )
    vocabulary_csv = tmp_path / "vocabulary.csv"
    vocabulary.to_csv(vocabulary_csv, index=False)
    return str(metadata_csv), str(vocabulary_csv), metadata


def test_make_splits_grouped_and_deterministic(tmp_path: Path) -> None:
    metadata_csv, vocabulary_csv, metadata = _synthetic_metadata(tmp_path)

    splits = make_splits(metadata_csv, vocabulary_csv, val_frac=0.2, seed=42)

    assert splits["seed"] == 42
    assert splits["group_by"] == "organisation"
    assert splits["label_list"] == sorted(metadata["word"].unique().tolist())

    train_ids, val_ids = set(splits["train"]), set(splits["val"])
    assert train_ids and val_ids
    assert not train_ids & val_ids
    assert train_ids | val_ids == set(metadata["clip_id"])

    org_of = dict(zip(metadata["clip_id"], metadata["organisation"]))
    train_orgs = {org_of[c] for c in train_ids}
    val_orgs = {org_of[c] for c in val_ids}
    assert not train_orgs & val_orgs

    frac = len(val_ids) / len(metadata)
    assert 0.1 <= frac <= 0.35

    again = make_splits(metadata_csv, vocabulary_csv, val_frac=0.2, seed=42)
    assert again == splits
