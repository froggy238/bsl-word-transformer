"""Per-frame skeleton normalisation.

Each frame is translated so the midpoint of the two shoulders is the
origin (all three coordinates) and scaled by the x-y Euclidean
shoulder-to-shoulder distance, making the representation invariant to
global translation and uniform rescaling. NaN coordinates propagate
unchanged (NaN in -> NaN out).
"""

from __future__ import annotations

import numpy as np

from .landmarks import LEFT_SHOULDER, N_COORDS, N_LANDMARKS, RIGHT_SHOULDER

EPS = 1e-6


def normalise_sequence(seq: np.ndarray) -> np.ndarray:
    """Normalise a landmark sequence frame by frame.

    Args:
        seq: Array of shape (T, 105, 3) with x, y, z coordinates; may
            contain NaN for missing landmarks.

    Returns:
        New array of shape (T, 105, 3): per frame, all coordinates are
        shifted by the shoulder midpoint and divided by
        max(xy shoulder distance, 1e-6). The input is not mutated.
    """
    seq = np.asarray(seq)
    if seq.ndim != 3 or seq.shape[1:] != (N_LANDMARKS, N_COORDS):
        raise ValueError(
            f"expected shape (T, {N_LANDMARKS}, {N_COORDS}), got {seq.shape}"
        )

    left = seq[:, LEFT_SHOULDER, :]
    right = seq[:, RIGHT_SHOULDER, :]
    origin = (left + right) / 2.0  # (T, 3)
    dist = np.sqrt(np.sum((left[:, :2] - right[:, :2]) ** 2, axis=-1))  # (T,)
    # np.maximum propagates NaN, so frames with missing shoulders stay NaN.
    scale = np.maximum(dist, EPS)

    out = (seq - origin[:, None, :]) / scale[:, None, None]
    return out.astype(seq.dtype, copy=False)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    demo = rng.uniform(0.0, 1.0, size=(5, N_LANDMARKS, N_COORDS)).astype(np.float32)
    demo[1, 33:54, :] = np.nan  # simulate a missing left hand in frame 1

    normed = normalise_sequence(demo)
    mid = (normed[:, LEFT_SHOULDER] + normed[:, RIGHT_SHOULDER]) / 2.0
    shoulder_xy = np.linalg.norm(
        normed[:, LEFT_SHOULDER, :2] - normed[:, RIGHT_SHOULDER, :2], axis=-1
    )
    print(f"input shape        : {demo.shape}, dtype {demo.dtype}")
    print(f"output shape       : {normed.shape}, dtype {normed.dtype}")
    print(f"shoulder midpoints : max |mid| = {np.abs(mid).max():.2e}")
    print(f"shoulder xy dist   : {shoulder_xy}")
    print(f"NaNs preserved     : {np.isnan(normed[1, 33:54]).all()}")
