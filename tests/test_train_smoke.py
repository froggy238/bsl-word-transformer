"""Smoke tests for the training loop on synthetic in-memory data.

Uses the real model/optimiser/scheduler wiring from src.train via
train_loop(run_dir=None), so no files or landmark data are needed.
"""

import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models import build_model
from src.train import make_scheduler, set_seed, train_loop

N_CLASSES = 4
SEQ_LEN = 64
IN_DIM = 315


def make_cfg(arch: str = "lstm", epochs: int = 5) -> dict:
    # Deliberately tiny model dims so the full loop runs in seconds on CPU, while
    # exercising the same config keys as the real 12-run experiment grid.
    return {
        "arch": arch,
        "epochs": epochs,
        "batch_size": 16,
        "lr": 3.0e-3,
        "weight_decay": 0.01,
        "warmup_epochs": 1,
        "label_smoothing": 0.1,
        "seq_len": SEQ_LEN,
        "in_dim": IN_DIM,
        "n_classes": N_CLASSES,
        "dropout": 0.0,
        "lstm_hidden": 32,
        "lstm_layers": 2,
        "d_model": 24,
        "n_layers": 1,
        "n_heads": 4,
        "d_ff": 48,
    }


def make_synthetic_loaders(
    n_train: int = 64,
    n_val: int = 32,
    batch_size: int = 16,
    noise: float = 0.05,
    data_seed: int = 0,
    loader_seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Class-separable random sequences: per-class mean pattern + small noise."""
    rng = np.random.default_rng(data_seed)
    # One fixed template per class; means[y] fancy-indexes to (n, SEQ_LEN, IN_DIM),
    # giving an easily separable problem where the loss must fall within a few epochs.
    means = rng.normal(0.0, 1.0, size=(N_CLASSES, SEQ_LEN, IN_DIM))

    def make(n: int) -> TensorDataset:
        y = rng.integers(0, N_CLASSES, size=n)
        x = (means[y] + noise * rng.normal(size=(n, SEQ_LEN, IN_DIM))).astype(
            np.float32
        )
        return TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.int64)))

    train_ds = make(n_train)
    val_ds = make(n_val)
    # A dedicated torch.Generator pins the shuffle order for run-to-run determinism.
    generator = torch.Generator()
    generator.manual_seed(loader_seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def test_train_loss_decreases() -> None:
    # End-to-end learning signal through the real optimiser/scheduler wiring: five
    # epochs on separable data must reduce the training loss and stay finite.
    set_seed(42)
    cfg = make_cfg("lstm", epochs=5)
    train_loader, val_loader = make_synthetic_loaders()
    model = build_model(cfg)
    result = train_loop(model, train_loader, val_loader, cfg, run_dir=None)
    hist = result["history"]
    assert len(hist) == 5
    assert all(math.isfinite(h["train_loss"]) for h in hist)
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    assert result["best_epoch"] >= 1


def _run_once() -> tuple[float, float]:
    set_seed(123)
    cfg = make_cfg("lstm", epochs=2)
    train_loader, val_loader = make_synthetic_loaders(data_seed=7, loader_seed=123)
    model = build_model(cfg)
    result = train_loop(model, train_loader, val_loader, cfg, run_dir=None)
    last = result["history"][-1]
    return last["train_loss"], last["val_loss"]

def test_two_seeded_runs_identical() -> None:
    # Whole-run reproducibility: set_seed must control every RNG (torch, numpy,
    # loader shuffling) — the dissertation's seeds 42/43/44 grid depends on this.
    loss_a = _run_once()
    loss_b = _run_once()
    assert abs(loss_a[0] - loss_b[0]) < 1e-6
    assert abs(loss_a[1] - loss_b[1]) < 1e-6


def test_transformer_wiring_runs() -> None:
    # One-epoch smoke test: the transformer path builds and trains without shape
    # or device errors (accuracy is not the point here).
    set_seed(0)
    cfg = make_cfg("transformer", epochs=1)
    train_loader, val_loader = make_synthetic_loaders(n_train=32, n_val=16)
    model = build_model(cfg)
    result = train_loop(model, train_loader, val_loader, cfg, run_dir=None)
    assert math.isfinite(result["history"][-1]["train_loss"])
    assert 0.0 <= result["best_val_acc"] <= 1.0


def test_scheduler_warmup_then_cosine() -> None:
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    scheduler = make_scheduler(optimizer, epochs=10, warmup_epochs=5)
    # Inspect the LambdaLR multiplier directly: linear warm-up 1/5..5/5 over epochs
    # 0-4, then cosine decay — warm-up avoids destabilising early transformer steps.
    lam = scheduler.lr_lambdas[0]
    assert [lam(e) for e in range(5)] == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])
    assert lam(5) == pytest.approx(1.0)  # cosine peak right after warm-up
    factors = [lam(e) for e in range(5, 10)]
    assert all(a >= b for a, b in zip(factors, factors[1:]))  # monotone cosine decay
    # Final factor equals the cosine formula at 4/5 of the post-warm-up span.
    assert lam(9) == pytest.approx(0.5 * (1.0 + math.cos(math.pi * 4 / 5)))
