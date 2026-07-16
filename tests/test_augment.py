"""Tests for src.augment. Synthetic arrays only; fully seeded/deterministic."""

from __future__ import annotations

import numpy as np
import pytest

from src.augment import (
    JITTER_SIGMA,
    MIN_FRAMES,
    ROTATION_MAX_DEG,
    SCALE_RANGE,
    SPEED_RANGE,
    TRANSLATE_RANGE,
    spatial_augment,
    temporal_augment,
)


class ForcedGateRng:
    """Duck-typed Generator stand-in with scripted gate outcomes.

    Each ``random()`` call pops the next scripted boolean: True -> 0.25
    (gate fires, since gates test ``< 0.5``), False -> 0.75. Parameter
    draws (uniform/normal/choice) delegate to a real seeded Generator.
    """

    def __init__(self, gates: list[bool], seed: int = 0) -> None:
        self._gates = list(gates)
        self._inner = np.random.default_rng(seed)

    def random(self) -> float:
        assert self._gates, "spatial/temporal augment drew more gates than scripted"
        return 0.25 if self._gates.pop(0) else 0.75

    def uniform(self, low=0.0, high=1.0, size=None):
        return self._inner.uniform(low, high, size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        return self._inner.normal(loc, scale, size)

    def choice(self, a, size=None, replace=True):
        return self._inner.choice(a, size=size, replace=replace)


def make_seq(t: int = 64, seed: int = 1, min_radius: float = 0.0) -> np.ndarray:
    """Random (t, 105, 3) float32 sequence; optionally keep xy off-origin."""
    rng = np.random.default_rng(seed)
    seq = rng.uniform(-1.0, 1.0, size=(t, 105, 3)).astype(np.float32)
    if min_radius > 0.0:
        xy = seq[..., :2]
        r = np.linalg.norm(xy, axis=-1, keepdims=True)
        scale = np.maximum(min_radius / np.maximum(r, 1e-9), 1.0)
        seq[..., :2] = (xy * scale).astype(np.float32)
    return seq


# ---------------------------------------------------------------------------
# spatial_augment
# ---------------------------------------------------------------------------


def test_spatial_shape_dtype_finite() -> None:
    seq = make_seq()
    for seed in range(50):
        out = spatial_augment(seq, np.random.default_rng(seed))
        assert out.shape == (64, 105, 3)
        assert out.dtype == np.float32
        assert np.all(np.isfinite(out))


def test_spatial_identity_when_no_transform_fires() -> None:
    seq = make_seq()
    out = spatial_augment(seq, ForcedGateRng([False, False, False, False]))
    assert np.array_equal(out, seq)
    assert not np.shares_memory(out, seq)


def test_spatial_rotation_only_bounded_angle() -> None:
    seq = make_seq(seed=2, min_radius=0.2)
    max_theta = np.deg2rad(ROTATION_MAX_DEG)
    for seed in range(20):
        out = spatial_augment(seq, ForcedGateRng([True, False, False, False], seed=seed))
        # z untouched by an xy-plane rotation
        assert np.array_equal(out[..., 2], seq[..., 2])
        # xy norms preserved
        r_in = np.linalg.norm(seq[..., :2], axis=-1)
        r_out = np.linalg.norm(out[..., :2], axis=-1)
        assert np.allclose(r_out, r_in, rtol=1e-4, atol=1e-5)
        # every point rotated by the same angle, |angle| <= 15 degrees
        diff = np.arctan2(out[..., 1], out[..., 0]) - np.arctan2(seq[..., 1], seq[..., 0])
        diff = (diff + np.pi) % (2.0 * np.pi) - np.pi
        assert np.max(np.abs(diff)) <= max_theta + 5e-3
        assert np.ptp(diff) < 2e-3


def test_spatial_scale_only_uniform_and_bounded() -> None:
    seq = make_seq(seed=3)
    for seed in range(20):
        out = spatial_augment(seq, ForcedGateRng([False, True, False, False], seed=seed))
        mask = np.abs(seq) > 1e-3
        ratios = out[mask] / seq[mask]
        assert SCALE_RANGE[0] - 1e-4 <= ratios.min()
        assert ratios.max() <= SCALE_RANGE[1] + 1e-4
        assert np.ptp(ratios) < 1e-3  # single uniform factor for all coords


def test_spatial_translate_only_constant_xy() -> None:
    seq = make_seq(seed=4)
    for seed in range(20):
        out = spatial_augment(seq, ForcedGateRng([False, False, True, False], seed=seed))
        assert np.array_equal(out[..., 2], seq[..., 2])
        dx = out[..., 0] - seq[..., 0]
        dy = out[..., 1] - seq[..., 1]
        for d in (dx, dy):
            assert np.ptp(d) < 1e-5  # constant per sequence
            assert TRANSLATE_RANGE[0] - 1e-4 <= d.mean() <= TRANSLATE_RANGE[1] + 1e-4


def test_spatial_jitter_only_statistics() -> None:
    seq = make_seq(seed=5)
    out = spatial_augment(seq, ForcedGateRng([False, False, False, True], seed=6))
    diff = (out - seq).ravel()
    assert abs(diff.mean()) < 3e-4
    assert diff.std() == pytest.approx(JITTER_SIGMA, rel=0.1)
    assert np.max(np.abs(diff)) < 6.0 * JITTER_SIGMA


def test_spatial_bounded_envelope_over_many_draws() -> None:
    # rotation preserves xy norm; with |coord| <= 1 the composition is
    # bounded by 1.1 * sqrt(2) + 0.05 + 6 sigma in x/y (less in z).
    seq = make_seq(seed=7)
    bound = 1.1 * np.sqrt(2.0) + abs(TRANSLATE_RANGE[1]) + 6.0 * JITTER_SIGMA
    for seed in range(200):
        out = spatial_augment(seq, np.random.default_rng(seed))
        assert np.all(np.isfinite(out))
        assert np.max(np.abs(out)) <= bound


# ---------------------------------------------------------------------------
# temporal_augment
# ---------------------------------------------------------------------------


def test_temporal_length_bounds() -> None:
    t = 50
    seq = make_seq(t=t, seed=8)
    # speed resample gives round(t / f), f in (0.8, 1.2); frame drop removes
    # at most floor(0.1 * t) frames; otherwise unchanged.
    lo = int(np.floor(t / SPEED_RANGE[1])) - 1
    hi = int(np.ceil(t / SPEED_RANGE[0])) + 1
    for seed in range(200):
        out = temporal_augment(seq, np.random.default_rng(seed))
        assert out.shape[1:] == (105, 3)
        assert out.dtype == np.float32
        assert MIN_FRAMES <= out.shape[0]
        assert lo <= out.shape[0] <= hi


def test_temporal_never_below_min_frames() -> None:
    for t in (8, 9, 12):
        seq = make_seq(t=t, seed=9)
        for seed in range(30):
            out = temporal_augment(seq, np.random.default_rng(seed))
            assert out.shape[0] >= min(t, MIN_FRAMES)
        for branch in (True, False):  # forced speed / forced drop
            for seed in range(10):
                out = temporal_augment(seq, ForcedGateRng([True, branch], seed=seed))
                assert out.shape[0] >= MIN_FRAMES


def test_temporal_noop_returns_equal_copy() -> None:
    seq = make_seq(t=40, seed=10)
    out = temporal_augment(seq, ForcedGateRng([False]))
    assert np.array_equal(out, seq)
    assert not np.shares_memory(out, seq)


def test_temporal_frame_drop_preserves_order() -> None:
    t = 60
    seq = np.zeros((t, 105, 3), dtype=np.float32)
    seq[:, :, 0] = np.arange(t, dtype=np.float32)[:, None]  # time channel
    for seed in range(30):
        out = temporal_augment(seq, ForcedGateRng([True, False], seed=seed))
        times = out[:, 0, 0]
        assert np.all(np.diff(times) > 0)  # strictly increasing => order kept
        assert set(times.tolist()) <= set(range(t))  # frames taken verbatim
        assert t - int(t * 0.1) <= out.shape[0] <= t


def test_temporal_speed_resample_monotonic_time() -> None:
    t = 60
    seq = np.zeros((t, 105, 3), dtype=np.float32)
    seq[:, :, 0] = np.arange(t, dtype=np.float32)[:, None]
    for seed in range(30):
        out = temporal_augment(seq, ForcedGateRng([True, True], seed=seed))
        times = out[:, 0, 0]
        assert times[0] == pytest.approx(0.0, abs=1e-5)
        assert times[-1] == pytest.approx(t - 1, abs=1e-3)
        if out.shape[0] > 1:
            assert np.all(np.diff(times) > 0)


# ---------------------------------------------------------------------------
# determinism and purity
# ---------------------------------------------------------------------------


def test_same_seed_gives_identical_output() -> None:
    seq = make_seq(t=50, seed=11)
    for fn in (spatial_augment, temporal_augment):
        a = fn(seq, np.random.default_rng(123))
        b = fn(seq, np.random.default_rng(123))
        assert np.array_equal(a, b)


def test_input_not_mutated() -> None:
    seq = make_seq(t=50, seed=12)
    ref = seq.copy()
    for seed in range(20):
        spatial_augment(seq, np.random.default_rng(seed))
        temporal_augment(seq, np.random.default_rng(seed))
    assert np.array_equal(seq, ref)


def test_float64_input_returns_float32() -> None:
    seq = make_seq(t=32, seed=13).astype(np.float64)
    assert spatial_augment(seq, np.random.default_rng(0)).dtype == np.float32
    assert temporal_augment(seq, np.random.default_rng(0)).dtype == np.float32
