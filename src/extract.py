"""MediaPipe Holistic landmark extraction and short-gap interpolation.

``fill_gaps`` depends only on numpy. mediapipe, cv2, pandas and tqdm are
imported lazily so this module imports cleanly without them; a clear
RuntimeError is raised only when extraction is actually attempted.

CLI:
    python -m src.extract --videos data/raw_videos --out data/landmarks
    python -m src.extract --videos data/test_videos --out data/test_landmarks
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from src.landmarks import (
    LEFT_HAND_SLICE,
    MOUTH_FACE_INDICES,
    MOUTH_SLICE,
    N_COORDS,
    N_LANDMARKS,
    POSE_SLICE,
    RIGHT_HAND_SLICE,
)

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")

# (landmark block, presence column) pairs used for gap filling.
_BLOCKS: tuple[tuple[slice, int], ...] = (
    (LEFT_HAND_SLICE, 0),
    (RIGHT_HAND_SLICE, 1),
    (MOUTH_SLICE, 2),
)


def _missing_runs(present: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) index ranges where ``present`` is False."""
    runs: list[tuple[int, int]] = []
    t, n = 0, present.shape[0]
    while t < n:
        if not present[t]:
            start = t
            while t < n and not present[t]:
                t += 1
            runs.append((start, t))
        else:
            t += 1
    return runs


def fill_gaps(
    seq: np.ndarray, presence: np.ndarray, max_gap: int = 5
) -> np.ndarray:
    """Linearly interpolate short missing runs per hand/mouth block.

    Runs of <= ``max_gap`` consecutive missing frames that are bounded by
    detected frames on both sides are linearly interpolated. Longer runs
    and leading/trailing missing runs are left NaN.

    Args:
        seq: (T, 105, 3) landmarks with NaN where a block was undetected.
        presence: (T, 3) columns [left_hand, right_hand, face], 1.0/0.0.
        max_gap: longest run length (frames) that gets interpolated.

    Returns:
        A new (T, 105, 3) array; the input is not mutated.
    """
    out = seq.copy()
    n_frames = seq.shape[0]
    for block, col in _BLOCKS:
        present = presence[:, col] > 0.5
        for start, end in _missing_runs(present):
            if start == 0 or end == n_frames:
                continue  # leading/trailing gap: no anchor on one side
            if end - start > max_gap:
                continue
            prev, nxt = start - 1, end
            span = nxt - prev
            for t in range(start, end):
                alpha = (t - prev) / span
                out[t, block] = (
                    (1.0 - alpha) * out[prev, block] + alpha * out[nxt, block]
                )
    return out


def _landmark_array(landmarks) -> np.ndarray:
    """Convert a MediaPipe landmark list to an (N, 3) float32 array."""
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32
    )


def extract_video(path: str, holistic) -> tuple[np.ndarray, np.ndarray, float]:
    """Run MediaPipe Holistic over a video file.

    Args:
        path: video file path.
        holistic: an initialised ``mediapipe.solutions.holistic.Holistic``.

    Returns:
        (landmarks (T, 105, 3) float32 with NaN for undetected blocks,
         presence (T, 3) float32 [left_hand, right_hand, face],
         fps).
    """
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    frames: list[np.ndarray] = []
    pres: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)

            lm = np.full((N_LANDMARKS, N_COORDS), np.nan, dtype=np.float32)
            p = np.zeros(3, dtype=np.float32)
            if results.pose_landmarks is not None:
                lm[POSE_SLICE] = _landmark_array(
                    results.pose_landmarks.landmark
                )
            if results.left_hand_landmarks is not None:
                lm[LEFT_HAND_SLICE] = _landmark_array(
                    results.left_hand_landmarks.landmark
                )
                p[0] = 1.0
            if results.right_hand_landmarks is not None:
                lm[RIGHT_HAND_SLICE] = _landmark_array(
                    results.right_hand_landmarks.landmark
                )
                p[1] = 1.0
            if results.face_landmarks is not None:
                face = _landmark_array(results.face_landmarks.landmark)
                lm[MOUTH_SLICE] = face[MOUTH_FACE_INDICES]
                p[2] = 1.0
            frames.append(lm)
            pres.append(p)
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return np.stack(frames), np.stack(pres), float(fps)


def _create_holistic():
    """Create a Holistic instance; raise clearly if mediapipe is missing."""
    try:
        import mediapipe as mp
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "mediapipe is required for landmark extraction "
            "(pip install mediapipe==0.10.9)"
        ) from exc
    return mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=1
    )


def hand_dropout_rate(landmarks: np.ndarray, presence: np.ndarray) -> float:
    """Fraction of frames with either hand missing while pose is present."""
    pose_present = np.isfinite(landmarks[:, POSE_SLICE]).any(axis=(1, 2))
    either_missing = (presence[:, 0] < 0.5) | (presence[:, 1] < 0.5)
    denom = int(pose_present.sum())
    if denom == 0:
        return float("nan")
    return float((pose_present & either_missing).sum() / denom)


def _clip_source(clip_id: str) -> str:
    """Fallback source token from a {word}_{source}_{nnn} clip id."""
    parts = clip_id.split("_")
    return parts[-2] if len(parts) >= 3 else "unknown"


def _organisation_map(metadata_csv: str) -> dict[str, str]:
    """clip_id -> organisation from metadata.csv (empty if unavailable)."""
    if not Path(metadata_csv).exists():
        return {}
    import pandas as pd

    try:
        df = pd.read_csv(metadata_csv, dtype=str)
    except Exception as exc:
        print(f"Warning: could not read {metadata_csv}: {exc}")
        return {}
    if "clip_id" not in df.columns or "organisation" not in df.columns:
        return {}
    return dict(zip(df["clip_id"], df["organisation"].fillna("unknown")))


def _print_dropout_summary(out_dir: Path, metadata_csv: str) -> None:
    """Print mean hand-dropout rate per source organisation over all npz."""
    org_map = _organisation_map(metadata_csv)
    per_org: dict[str, list[float]] = {}
    for npz_path in sorted(out_dir.glob("*/*.npz")):
        clip_id = npz_path.stem
        try:
            with np.load(npz_path) as data:
                rate = hand_dropout_rate(data["landmarks"], data["presence"])
        except Exception as exc:
            print(f"Warning: could not read {npz_path}: {exc}")
            continue
        org = org_map.get(clip_id, _clip_source(clip_id))
        per_org.setdefault(org, []).append(rate)
    if not per_org:
        print("No cached landmark files found; no dropout summary.")
        return
    print("\nHand dropout rate by source organisation")
    print(f"{'organisation':<32}{'clips':>8}{'mean dropout':>16}")
    for org in sorted(per_org):
        rates = np.array(per_org[org], dtype=np.float64)
        mean = float(np.nanmean(rates)) if np.isfinite(rates).any() else float("nan")
        print(f"{org:<32}{len(rates):>8}{mean:>16.3f}")


def _find_videos(videos_dir: Path) -> list[Path]:
    """All {word}/{clip}.<ext> videos under ``videos_dir``, sorted."""
    videos: list[Path] = []
    for word_dir in sorted(p for p in videos_dir.iterdir() if p.is_dir()):
        for f in sorted(word_dir.iterdir()):
            if f.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(f)
    return videos


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe Holistic landmarks from videos."
    )
    parser.add_argument("--videos", default="data/raw_videos",
                        help="input dir of {word}/{clip}.mp4 folders")
    parser.add_argument("--out", default="data/landmarks",
                        help="output dir for {word}/{clip_id}.npz files")
    parser.add_argument("--metadata", default="data/metadata.csv",
                        help="metadata csv for the dropout summary join")
    args = parser.parse_args(argv)

    from tqdm import tqdm

    videos_dir = Path(args.videos)
    out_dir = Path(args.out)
    if not videos_dir.is_dir():
        print(f"Error: videos directory not found: {videos_dir}")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = _find_videos(videos_dir)
    todo = [
        v for v in videos
        if not (out_dir / v.parent.name / f"{v.stem}.npz").exists()
    ]
    print(f"Found {len(videos)} videos ({len(videos) - len(todo)} cached, "
          f"{len(todo)} to extract)")

    n_ok, n_fail = 0, 0
    for video in tqdm(todo, desc="Extracting", unit="clip"):
        npz_path = out_dir / video.parent.name / f"{video.stem}.npz"
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Fresh Holistic per clip: clips are independent, so tracking
            # state must not carry over between them.
            with _create_holistic() as holistic:
                landmarks, presence, fps = extract_video(str(video), holistic)
            # Temp name must keep the .npz suffix or savez appends one.
            tmp_path = npz_path.with_name(npz_path.stem + ".tmp.npz")
            np.savez_compressed(
                tmp_path,
                landmarks=landmarks.astype(np.float32),
                presence=presence.astype(np.float32),
                fps=np.float32(fps),
            )
            os.replace(tmp_path, npz_path)
            n_ok += 1
        except RuntimeError:
            raise  # mediapipe missing: fail fast rather than per clip
        except Exception as exc:
            n_fail += 1
            print(f"\nFailed on {video}: {exc}")

    print(f"Extracted {n_ok} clips, {n_fail} failures.")
    _print_dropout_summary(out_dir, args.metadata)
    return 0


if __name__ == "__main__":
    sys.exit(main())
