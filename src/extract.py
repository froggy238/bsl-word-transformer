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
# Pose has no presence column: only the fragile blocks (hands, mouth) drop in
# and out frame-to-frame, so only they are candidates for interpolation.
_BLOCKS: tuple[tuple[slice, int], ...] = (
    (LEFT_HAND_SLICE, 0),
    (RIGHT_HAND_SLICE, 1),
    (MOUTH_SLICE, 2),
)


def _missing_runs(present: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) index ranges where ``present`` is False."""
    runs: list[tuple[int, int]] = []
    t, n = 0, present.shape[0]
    # Single left-to-right scan: on hitting a missing frame, remember where it
    # started, then advance the same cursor past the whole run. Each frame is
    # visited once (O(T)) and the returned ranges can never overlap.
    while t < n:
        if not present[t]:
            start = t
            while t < n and not present[t]:
                t += 1  # inner loop consumes the entire missing run
            runs.append((start, t))  # half-open: t is the first detected frame (or n)
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
    out = seq.copy()  # work on a copy so cached arrays are never mutated
    n_frames = seq.shape[0]
    # Blocks are filled independently: the left hand can drop out while the
    # right hand and mouth are still tracked, and vice versa.
    for block, col in _BLOCKS:
        present = presence[:, col] > 0.5  # (T,) float 1.0/0.0 column -> boolean mask
        for start, end in _missing_runs(present):
            if start == 0 or end == n_frames:
                continue  # leading/trailing gap: no anchor on one side
            if end - start > max_gap:
                continue  # long occlusion: interpolating would invent motion, leave NaN
            # Anchors: last detected frame before the gap, first one after it.
            prev, nxt = start - 1, end
            span = nxt - prev  # gap length + 1 = number of interpolation steps
            for t in range(start, end):
                # alpha rises 1/span .. (span-1)/span across the gap, so the
                # straight-line blend meets both anchor frames exactly.
                alpha = (t - prev) / span
                out[t, block] = (
                    (1.0 - alpha) * out[prev, block] + alpha * out[nxt, block]
                )  # convex blend broadcast over the whole (block_len, 3) slice
    return out


def _landmark_array(landmarks) -> np.ndarray:
    """Convert a MediaPipe landmark list to an (N, 3) float32 array."""
    # MediaPipe coordinates: x/y are normalised to [0, 1] by image width and
    # height; z is a rough relative depth. The later shoulder-based
    # normalisation removes the dependence on image size anyway.
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
        fps = 25.0  # some containers report 0/NaN fps; fall back to PAL 25
    frames: list[np.ndarray] = []
    pres: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV is BGR; MediaPipe wants RGB
            rgb.flags.writeable = False  # read-only hint lets MediaPipe avoid copying the frame
            results = holistic.process(rgb)

            # Start each frame all-NaN with zero presence, then overwrite only
            # the blocks Holistic actually detected; the rest stay NaN.
            lm = np.full((N_LANDMARKS, N_COORDS), np.nan, dtype=np.float32)  # (105, 3)
            p = np.zeros(3, dtype=np.float32)  # [left_hand, right_hand, face] flags
            # Holistic returns None for each undetected block, hence the
            # per-block guards. Pose has no presence flag; a missing pose is
            # simply left as NaN.
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
                face = _landmark_array(results.face_landmarks.landmark)  # 468-point face mesh
                lm[MOUTH_SLICE] = face[MOUTH_FACE_INDICES]  # fancy-index the 30 mouth points
                p[2] = 1.0
            frames.append(lm)
            pres.append(p)
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    # Stack the per-frame lists into (T, 105, 3) landmarks and (T, 3) presence.
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
    # Video mode (static_image_mode=False) tracks landmarks between frames,
    # which is faster and smoother than detecting from scratch each frame;
    # model_complexity=1 trades a little accuracy for laptop-CPU speed, in
    # line with the real-time (>=15 fps end-to-end) criterion.
    return mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=1
    )


def hand_dropout_rate(landmarks: np.ndarray, presence: np.ndarray) -> float:
    """Fraction of frames with either hand missing while pose is present."""
    # Condition on the signer being visible at all: frames with no pose (e.g.
    # blank lead-in/out) would otherwise inflate the dropout rate.
    pose_present = np.isfinite(landmarks[:, POSE_SLICE]).any(axis=(1, 2))  # (T, 33, 3) -> (T,)
    either_missing = (presence[:, 0] < 0.5) | (presence[:, 1] < 0.5)  # (T,) per-frame flag
    denom = int(pose_present.sum())
    if denom == 0:
        return float("nan")
    return float((pose_present & either_missing).sum() / denom)


def _clip_source(clip_id: str) -> str:
    """Fallback source token from a {word}_{source}_{nnn} clip id."""
    parts = clip_id.split("_")
    # e.g. "hello_signbsl_003" -> "signbsl": the second-to-last token names
    # the source site when the clip is missing from metadata.csv.
    return parts[-2] if len(parts) >= 3 else "unknown"


def _organisation_map(metadata_csv: str) -> dict[str, str]:
    """clip_id -> organisation from metadata.csv (empty if unavailable)."""
    # Organisation is the grouping key behind the signer-independent
    # train/val split, so the dropout summary reports at that granularity.
    if not Path(metadata_csv).exists():
        return {}
    import pandas as pd

    try:
        df = pd.read_csv(metadata_csv, dtype=str)  # dtype=str stops ids being numeric-coerced
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
    # Join every cached npz back to its organisation: prefer the metadata.csv
    # mapping, else fall back to the source token parsed from the clip id.
    for npz_path in sorted(out_dir.glob("*/*.npz")):  # {word}/{clip_id}.npz layout
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
        # nanmean skips clips whose rate is NaN (pose never detected); the
        # guard avoids numpy's all-NaN RuntimeWarning when none are finite.
        mean = float(np.nanmean(rates)) if np.isfinite(rates).any() else float("nan")
        print(f"{org:<32}{len(rates):>8}{mean:>16.3f}")


def _find_videos(videos_dir: Path) -> list[Path]:
    """All {word}/{clip}.<ext> videos under ``videos_dir``, sorted."""
    videos: list[Path] = []
    # Sorted at both levels so extraction order is deterministic across runs
    # and operating systems (reproducibility of logs and progress counts).
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
    # Resumable cache: a clip is skipped iff its final npz exists. The atomic
    # rename below guarantees any existing file is complete, never truncated.
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
            # (Video-mode tracking would otherwise seed a clip's first frames
            # from the last frames of the previous, unrelated clip.)
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
            # os.replace is atomic on the same filesystem, so a crash or kill
            # can never leave a half-written npz that the cache would trust.
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
