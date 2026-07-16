"""Data augmentation for pose-landmark sequences.

Spatial transforms operate on fixed-length normalised skeletons of shape
(64, 105, 3); temporal transforms operate on variable-length sequences
BEFORE the fixed 64-frame resampling. All functions are pure (the input
array is never mutated) and return float32 arrays.

Coordinate convention: axis -1 holds (x, y, z). Rotation is about the
origin in the x-y plane only; translation shifts x and y only; scale and
jitter apply to all three coordinates.
"""

from __future__ import annotations

import numpy as np

ROTATION_MAX_DEG: float = 15.0
SCALE_RANGE: tuple[float, float] = (0.9, 1.1)
TRANSLATE_RANGE: tuple[float, float] = (-0.05, 0.05)
JITTER_SIGMA: float = 0.01
SPEED_RANGE: tuple[float, float] = (0.8, 1.2)
MAX_FRAME_DROP_FRAC: float = 0.10
MIN_FRAMES: int = 8

_TRANSFORM_P: float = 0.5


def _resample(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resample ``seq`` along the time axis to ``target_len`` frames.

    Local copy (not imported from src.dataset) to avoid a circular import.
    """
    t = seq.shape[0]
    if t == target_len:
        return seq.astype(np.float32, copy=True)
    if t == 1:
        return np.repeat(seq, target_len, axis=0).astype(np.float32)
    pos = np.linspace(0.0, float(t - 1), target_len)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, t - 1)
    w = (pos - lo).astype(np.float32).reshape(-1, *([1] * (seq.ndim - 1)))
    out = (1.0 - w) * seq[lo].astype(np.float32) + w * seq[hi].astype(np.float32)
    return out.astype(np.float32)


def spatial_augment(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomly perturb a skeleton sequence in space.

    Each of the four transforms fires independently with p=0.5, in a fixed
    order (rotate, scale, translate, jitter):

    - rotation about the origin in the x-y plane by U(-15, +15) degrees
      (z untouched);
    - uniform scale of all coordinates by U(0.9, 1.1);
    - translation of x and y by per-sequence constants U(-0.05, 0.05);
    - Gaussian jitter N(0, 0.01) added to every coordinate.

    Args:
        seq: array of shape (T, 105, 3), typically (64, 105, 3).
        rng: numpy random Generator; parameter draws only occur for
            transforms whose gate fires, so draw order is deterministic.

    Returns:
        New float32 array with the same shape as ``seq``.
    """
    out = np.asarray(seq, dtype=np.float32).copy()

    if rng.random() < _TRANSFORM_P:
        theta = np.deg2rad(rng.uniform(-ROTATION_MAX_DEG, ROTATION_MAX_DEG))
        cos_t = np.float32(np.cos(theta))
        sin_t = np.float32(np.sin(theta))
        x = out[..., 0].copy()
        y = out[..., 1].copy()
        out[..., 0] = cos_t * x - sin_t * y
        out[..., 1] = sin_t * x + cos_t * y

    if rng.random() < _TRANSFORM_P:
        out *= np.float32(rng.uniform(*SCALE_RANGE))

    if rng.random() < _TRANSFORM_P:
        out[..., 0] += np.float32(rng.uniform(*TRANSLATE_RANGE))
        out[..., 1] += np.float32(rng.uniform(*TRANSLATE_RANGE))

    if rng.random() < _TRANSFORM_P:
        out += rng.normal(0.0, JITTER_SIGMA, size=out.shape).astype(np.float32)

    return out


def temporal_augment(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomly perturb sequence timing before the fixed 64-frame resample.

    With p=0.5 no perturbation is applied (a copy is returned). Otherwise
    one of two perturbations is chosen uniformly:

    - speed resample: linear interpolation to round(T / f) frames with
      f ~ U(0.8, 1.2);
    - frame drop: remove floor(T * d) random frames, d ~ U(0, 0.10),
      preserving the order of the retained frames.

    The result never has fewer than ``MIN_FRAMES`` (8) frames.

    Args:
        seq: array of shape (T, 105, 3).
        rng: numpy random Generator.

    Returns:
        New float32 array of shape (T', 105, 3).
    """
    src = np.asarray(seq, dtype=np.float32)
    if rng.random() >= _TRANSFORM_P:
        return src.copy()

    t = src.shape[0]
    if rng.random() < 0.5:
        factor = rng.uniform(*SPEED_RANGE)
        new_len = max(MIN_FRAMES, int(round(t / factor)))
        return _resample(src, new_len)

    frac = rng.uniform(0.0, MAX_FRAME_DROP_FRAC)
    n_drop = min(int(np.floor(t * frac)), max(t - MIN_FRAMES, 0))
    if n_drop == 0:
        return src.copy()
    drop = rng.choice(t, size=n_drop, replace=False)
    keep = np.setdiff1d(np.arange(t), drop)
    return src[keep].copy()
