"""Tests for src.normalise.normalise_sequence (synthetic data only)."""

import numpy as np
import pytest

from src.landmarks import LEFT_HAND_SLICE, LEFT_SHOULDER, RIGHT_SHOULDER
from src.normalise import normalise_sequence

T = 10
ATOL = 1e-5


def _random_skeleton(seed: int) -> np.ndarray:
    """Seeded random (10, 105, 3) skeleton with well-separated shoulders."""
    rng = np.random.default_rng(seed)
    seq = rng.uniform(0.0, 1.0, size=(T, 105, 3))
    seq[:, LEFT_SHOULDER, 0] = 0.35 + 0.02 * rng.standard_normal(T)
    seq[:, RIGHT_SHOULDER, 0] = 0.65 + 0.02 * rng.standard_normal(T)
    return seq


def test_translation_invariance() -> None:
    seq = _random_skeleton(42)
    shift = np.array([0.3, -0.2, 0.15])
    np.testing.assert_allclose(
        normalise_sequence(seq + shift),
        normalise_sequence(seq),
        rtol=0.0,
        atol=ATOL,
    )


@pytest.mark.parametrize("s", [0.25, 3.0])
def test_uniform_scale_invariance(s: float) -> None:
    seq = _random_skeleton(43)
    np.testing.assert_allclose(
        normalise_sequence(seq * s),
        normalise_sequence(seq),
        rtol=0.0,
        atol=ATOL,
    )


def test_combined_translate_and_scale_invariance() -> None:
    seq = _random_skeleton(44)
    shift = np.array([-0.4, 0.25, 0.1])
    np.testing.assert_allclose(
        normalise_sequence(seq * 1.7 + shift),
        normalise_sequence(seq),
        rtol=0.0,
        atol=ATOL,
    )


def test_nan_propagation() -> None:
    seq = _random_skeleton(45)
    seq[2:6, LEFT_HAND_SLICE, :] = np.nan
    out = normalise_sequence(seq)
    nan_in = np.isnan(seq)
    assert np.array_equal(np.isnan(out), nan_in)
    assert np.all(np.isfinite(out[~nan_in]))


def test_canonical_frame() -> None:
    out = normalise_sequence(_random_skeleton(46))
    midpoint = (out[:, LEFT_SHOULDER] + out[:, RIGHT_SHOULDER]) / 2.0
    np.testing.assert_allclose(midpoint, 0.0, rtol=0.0, atol=ATOL)
    dist = np.linalg.norm(
        out[:, LEFT_SHOULDER, :2] - out[:, RIGHT_SHOULDER, :2], axis=-1
    )
    np.testing.assert_allclose(dist, 1.0, rtol=0.0, atol=ATOL)


def test_input_not_mutated() -> None:
    seq = _random_skeleton(47)
    seq[3, LEFT_HAND_SLICE, :] = np.nan
    before = seq.copy()
    normalise_sequence(seq)
    # assert_array_equal treats NaNs in matching positions as equal.
    np.testing.assert_array_equal(seq, before)


def test_shape_and_dtype_preserved() -> None:
    seq = _random_skeleton(48).astype(np.float32)
    out = normalise_sequence(seq)
    assert out.shape == (T, 105, 3)
    assert out.dtype == np.float32


def test_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        normalise_sequence(np.zeros((T, 104, 3)))
