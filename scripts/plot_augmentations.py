"""Visual sanity check for the augmentation pipeline.

Renders a side-by-side figure: the original (normalised) skeleton next to
each spatial augmentation applied in isolation, plus a temporal-augmentation
trajectory panel. Works from a synthetic skeleton by default, or a real
cached landmark file via --npz.

Usage:
    python scripts/plot_augmentations.py
    python scripts/plot_augmentations.py --npz data/landmarks/hello/hello_signbsl_001.npz
    python scripts/plot_augmentations.py --seed 7 --out results/figures/augmentation_check.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.augment import spatial_augment, temporal_augment  # noqa: E402

try:  # shared contract constants (fall back so the script runs standalone)
    from src.landmarks import LEFT_SHOULDER, POSE_SLICE, RIGHT_SHOULDER
except ImportError:  # pragma: no cover
    POSE_SLICE = slice(0, 33)
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

LEFT_WRIST = 15
RIGHT_WRIST = 16

# Subset of MediaPipe pose connections: arms, torso, legs.
POSE_CONNECTIONS: list[tuple[int, int]] = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (28, 30), (30, 32),
]


class SingleTransformRng:
    """Generator stand-in that fires exactly one scripted augmentation gate."""

    def __init__(self, gates: list[bool], seed: int) -> None:
        self._gates = list(gates)
        self._inner = np.random.default_rng(seed)

    def random(self) -> float:
        return 0.25 if self._gates.pop(0) else 0.75

    def uniform(self, low=0.0, high=1.0, size=None):
        return self._inner.uniform(low, high, size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        return self._inner.normal(loc, scale, size)

    def choice(self, a, size=None, replace=True):
        return self._inner.choice(a, size=size, replace=replace)


def _normalise(seq: np.ndarray) -> np.ndarray:
    """Shoulder-midpoint origin + xy shoulder-distance scale, per frame."""
    try:
        from src.normalise import normalise_sequence

        return normalise_sequence(seq)
    except ImportError:  # pragma: no cover
        out = seq.astype(np.float32).copy()
        mid = 0.5 * (out[:, LEFT_SHOULDER] + out[:, RIGHT_SHOULDER])
        dist = np.linalg.norm(
            out[:, LEFT_SHOULDER, :2] - out[:, RIGHT_SHOULDER, :2], axis=-1
        )
        out -= mid[:, None, :]
        out /= np.maximum(dist, 1e-6)[:, None, None]
        return out


def synthetic_sequence(t: int = 48) -> np.ndarray:
    """Plausible 105-landmark stick figure in image coords, waving one arm."""
    pose_xy = {
        0: (0.50, 0.20),
        1: (0.52, 0.18), 2: (0.53, 0.18), 3: (0.54, 0.18),
        4: (0.48, 0.18), 5: (0.47, 0.18), 6: (0.46, 0.18),
        7: (0.56, 0.19), 8: (0.44, 0.19),
        9: (0.52, 0.23), 10: (0.48, 0.23),
        11: (0.62, 0.35), 12: (0.38, 0.35),
        13: (0.66, 0.50), 14: (0.34, 0.50),
        15: (0.63, 0.63), 16: (0.37, 0.63),
        17: (0.63, 0.66), 18: (0.37, 0.66),
        19: (0.62, 0.67), 20: (0.38, 0.67),
        21: (0.61, 0.65), 22: (0.39, 0.65),
        23: (0.58, 0.70), 24: (0.42, 0.70),
        25: (0.58, 0.85), 26: (0.42, 0.85),
        27: (0.575, 0.97), 28: (0.425, 0.97),
        29: (0.57, 0.985), 30: (0.43, 0.985),
        31: (0.60, 0.99), 32: (0.40, 0.99),
    }
    frame = np.zeros((105, 3), dtype=np.float32)
    for idx, (x, y) in pose_xy.items():
        frame[idx, 0], frame[idx, 1] = x, y

    # hand blocks: small clusters around each wrist
    ang = np.linspace(0.0, 2.0 * np.pi, 21, endpoint=False)
    radius = 0.018 + 0.006 * np.cos(3 * ang)
    for block_start, wrist in ((33, LEFT_WRIST), (54, RIGHT_WRIST)):
        frame[block_start : block_start + 21, 0] = (
            frame[wrist, 0] + radius * np.cos(ang)
        )
        frame[block_start : block_start + 21, 1] = (
            frame[wrist, 1] + 0.02 + radius * np.sin(ang)
        )

    # mouth block: 30 points on an ellipse below the nose
    ang_m = np.linspace(0.0, 2.0 * np.pi, 30, endpoint=False)
    frame[75:105, 0] = 0.50 + 0.025 * np.cos(ang_m)
    frame[75:105, 1] = 0.23 + 0.012 * np.sin(ang_m)

    seq = np.repeat(frame[None], t, axis=0)
    # animate: right arm (elbow, wrist, fingers + right-hand block) waves
    phase = np.sin(np.linspace(0.0, 2.0 * np.pi, t)).astype(np.float32)
    moving = [14, 16, 18, 20, 22] + list(range(54, 75))
    seq[:, moving, 0] -= 0.08 * phase[:, None]
    seq[:, moving, 1] -= 0.10 * np.abs(phase)[:, None]
    return seq.astype(np.float32)


def load_npz_sequence(path: str) -> np.ndarray:
    data = np.load(path)
    return np.asarray(data["landmarks"], dtype=np.float32)


def plot_skeleton(ax, frame: np.ndarray, title: str, ref: np.ndarray | None = None) -> None:
    """Scatter all 105 points and draw limb lines for the pose block."""
    if ref is not None:
        pose_ref = ref[POSE_SLICE]
        for a, b in POSE_CONNECTIONS:
            ax.plot(
                [pose_ref[a, 0], pose_ref[b, 0]],
                [pose_ref[a, 1], pose_ref[b, 1]],
                color="0.8", lw=1.0, zorder=1,
            )
        ax.scatter(ref[:, 0], ref[:, 1], s=3, color="0.8", zorder=1)
    pose = frame[POSE_SLICE]
    for a, b in POSE_CONNECTIONS:
        ax.plot(
            [pose[a, 0], pose[b, 0]],
            [pose[a, 1], pose[b, 1]],
            color="tab:blue", lw=1.5, zorder=2,
        )
    ax.scatter(frame[:, 0], frame[:, 1], s=4, color="tab:red", zorder=3)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # image convention: y grows downwards
    ax.tick_params(labelsize=7)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--npz", default=None, help="cached landmark .npz to use instead of the synthetic skeleton")
    parser.add_argument("--out", default="results/figures/augmentation_check.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame", type=int, default=None, help="frame index to draw (default: middle frame)")
    args = parser.parse_args()

    seq = load_npz_sequence(args.npz) if args.npz else synthetic_sequence()
    seq = np.nan_to_num(_normalise(seq)).astype(np.float32)
    t = seq.shape[0]
    frame_idx = args.frame if args.frame is not None else t // 2
    base = seq[frame_idx]

    # apply each spatial transform in isolation via a scripted gate sequence
    single = {
        "Rotation (±15°)": [True, False, False, False],
        "Scale (0.9–1.1)": [False, True, False, False],
        "Translate (±0.05)": [False, False, True, False],
        "Jitter (σ=0.01)": [False, False, False, True],
    }
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    plot_skeleton(axes[0, 0], base, f"Original (frame {frame_idx}/{t})")
    for ax, (title, gates) in zip(axes.flat[1:5], single.items()):
        aug = spatial_augment(seq, SingleTransformRng(gates, seed=args.seed))
        plot_skeleton(ax, aug[frame_idx], title, ref=base)

    # temporal panel: right-wrist x trajectory, original vs speed-resampled
    ax_t = axes[1, 2]
    aug_t = temporal_augment(seq, SingleTransformRng([True, True], seed=args.seed))
    ax_t.plot(np.linspace(0, 1, t), seq[:, RIGHT_WRIST, 0], label=f"original ({t} frames)")
    ax_t.plot(
        np.linspace(0, 1, aug_t.shape[0]),
        aug_t[:, RIGHT_WRIST, 0],
        "--",
        label=f"speed-resampled ({aug_t.shape[0]} frames)",
    )
    ax_t.set_title("Temporal: right-wrist x trajectory", fontsize=10)
    ax_t.set_xlabel("normalised time", fontsize=8)
    ax_t.set_ylabel("x (shoulder units)", fontsize=8)
    ax_t.legend(fontsize=7)
    ax_t.tick_params(labelsize=7)

    fig.suptitle("Augmentation sanity check (grey = original)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
