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

# Floor for the shoulder distance: prevents division by ~0 in degenerate frames
# where the detector places both shoulders at (nearly) the same point.
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

    # Pick one landmark across all frames: (T, 105, 3) -> (T, 3) per shoulder.
    left = seq[:, LEFT_SHOULDER, :]
    right = seq[:, RIGHT_SHOULDER, :]
    # Shoulder midpoint = per-frame origin. If either shoulder is NaN, the
    # origin is NaN and the subtraction below turns the whole frame NaN -- a
    # deliberate missing-frame signal (mapped to zeros later, after gap-filling).
    origin = (left + right) / 2.0  # (T, 3)
    # Scale = shoulder distance in the image plane only ([:, :2] keeps x, y):
    # MediaPipe's z estimate is far noisier than x/y, so it is excluded from the
    # scale (z coordinates are still shifted and divided like x and y).
    dist = np.sqrt(np.sum((left[:, :2] - right[:, :2]) ** 2, axis=-1))  # (T,)
    # np.maximum propagates NaN, so frames with missing shoulders stay NaN.
    scale = np.maximum(dist, EPS)

    # Broadcasting: origin (T, 3) -> (T, 1, 3) subtracts from all 105 landmarks;
    # scale (T,) -> (T, 1, 1) divides every coordinate. Afterwards each frame
    # has its shoulder midpoint at (0, 0, 0) and xy shoulder distance 1, so the
    # features are invariant to where the signer stands and to camera zoom --
    # key for generalising from scraped sources to the unseen held-out signer.
    out = (seq - origin[:, None, :]) / scale[:, None, None]
    # Keep the caller's dtype (float32 throughout the pipeline); copy=False
    # avoids a second allocation when the dtype already matches.
    return out.astype(seq.dtype, copy=False)


if __name__ == "__main__":
    # Smoke test: seeded random frames plus one deliberately missing hand block,
    # checking midpoint ~ 0, xy shoulder distance ~ 1 and NaN pass-through.
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
