"""Training entry point for BSL word-level classifiers.

CLI:
    python -m src.train --config configs/transformer_aug_s42.yaml
        [--overfit-batch] [--shuffle-labels] [--epochs N] [--device cpu|cuda]

Sanity modes:
    --overfit-batch   trains on a single fixed batch and asserts >=95% batch
                      accuracy within 200 steps (checks the model can learn).
    --shuffle-labels  permutes the training labels once (seeded); validation
                      accuracy should sit near chance (1/n_classes). The run
                      is written under '{run_id}_shufflelabels' so it never
                      clobbers a real run.
"""

from __future__ import annotations

import argparse
import csv
import math
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.models import build_model, count_parameters

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[1]  # src/ -> repo root (for git provenance)


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU + CUDA) and force deterministic cuDNN."""
    # Every RNG the pipeline touches must be seeded: `random` (python-level shuffles), numpy
    # (label permutation, augmentation) and torch on CPU + all CUDA devices (weight init,
    # dropout, DataLoader shuffling).
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN normally autotunes and may pick non-deterministic kernels; forcing the
    # deterministic path (and disabling benchmarking) makes runs exactly repeatable at a
    # small speed cost — reproducibility matters for the seed-averaged 12-run grid.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_scheduler(
    optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Per-epoch LR schedule: linear warm-up then cosine decay to zero."""

    def lr_lambda(epoch: int) -> float:
        # LambdaLR multiplies the base lr by this factor; `epoch` counts from 0.
        if epoch < warmup_epochs:
            # Warm-up: factor ramps 1/w, 2/w, ..., 1 so the first updates on a randomly
            # initialised model are small (large early steps can destabilise Transformers).
            return (epoch + 1) / max(1, warmup_epochs)
        # Cosine decay: as (epoch - warmup)/span runs 0 -> 1 the cos argument runs 0 -> pi,
        # so the factor falls smoothly from 1 to 0 with no step-change drops. max(1, ...)
        # guards division by zero when epochs == warmup_epochs.
        span = max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / span))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_dataloaders(
    cfg: dict, shuffle_labels: bool = False
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Build train/val DataLoaders from splits.json per the config.

    Returns (train_loader, val_loader, label_list). Class index = position of
    the clip's word in label_list. With shuffle_labels=True the training
    labels are permuted once (seeded with cfg['seed']).
    """
    # Imported lazily so src.train stays importable for in-memory smoke tests
    # even if the data-layer module is unavailable.
    from src.dataset import BSLDataset, clip_id_to_word, load_splits

    # splits.json was built grouped by source organisation, so no signer appears in both
    # train and val — the signer-independence guarantee lives in that file, not here.
    splits = load_splits(cfg["splits_file"])
    label_list: list[str] = splits["label_list"]
    label_to_idx = {word: i for i, word in enumerate(label_list)}

    metadata = None
    meta_path = Path(cfg.get("metadata_csv", ""))
    if meta_path.is_file():
        import pandas as pd

        metadata = pd.read_csv(meta_path)

    train_ids: list[str] = splits["train"]
    val_ids: list[str] = splits["val"]
    # Labels come from the clip id itself (it encodes the word), so labels and files can
    # never be misaligned by a separate labels file.
    train_labels = [label_to_idx[clip_id_to_word(c)] for c in train_ids]
    val_labels = [label_to_idx[clip_id_to_word(c)] for c in val_ids]

    if shuffle_labels:
        # Destroy the input->label association while keeping inputs and label frequencies
        # intact. A local seeded rng leaves the global RNG state untouched, so everything
        # else (init, shuffling, augmentation) matches the real run. If val accuracy still
        # beats chance afterwards, labels are leaking (e.g. duplicated or memorised clips).
        perm = np.random.default_rng(int(cfg["seed"])).permutation(len(train_labels))
        train_labels = [train_labels[i] for i in perm]
        print(
            "[train] --shuffle-labels: training labels permuted; "
            f"expect val accuracy near chance ({1.0 / cfg.get('n_classes', 50):.3f})"
        )

    seq_len = int(cfg.get("seq_len", 64))  # frames per clip after temporal resampling
    train_ds = BSLDataset(
        train_ids,
        train_labels,
        cfg["landmarks_dir"],
        metadata,
        augment=bool(cfg.get("augment", False)),
        seq_len=seq_len,
        seed=int(cfg["seed"]),
    )
    val_ds = BSLDataset(
        val_ids,
        val_labels,
        cfg["landmarks_dir"],
        metadata,
        augment=False,  # validation must measure a fixed distribution — never augmented
        seq_len=seq_len,
        seed=int(cfg["seed"]),
    )

    # Dedicated seeded Generator: the train loader's shuffle order is reproducible without
    # depending on (or perturbing) torch's global RNG stream.
    generator = torch.Generator()
    generator.manual_seed(int(cfg["seed"]))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=0,  # single-process loading: deterministic order, no Windows fork issues
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0
    )
    return train_loader, val_loader, label_list


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    progress: bool = False,
    desc: str = "",
) -> tuple[float, float]:
    """One pass over loader; trains if optimizer is given. Returns (loss, acc)."""
    training = optimizer is not None  # one function serves both train and eval passes
    model.train(training)  # toggles dropout etc. between train and eval behaviour
    total_loss, total_correct, total = 0.0, 0, 0
    iterator = loader
    if progress and tqdm is not None:
        iterator = tqdm(loader, desc=desc, leave=False)
    # Pick the autograd context at runtime: eval passes skip gradient bookkeeping (faster,
    # less memory) and cannot accidentally update weights.
    with torch.enable_grad() if training else torch.no_grad():
        for inputs, targets in iterator:
            # inputs (B, 64, 315) flattened landmark features; targets (B,) class indices
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            n = targets.size(0)
            # CrossEntropyLoss returns the batch mean, so re-weight by batch size before
            # summing — otherwise a smaller final batch would be over-weighted.
            total_loss += loss.item() * n
            total_correct += (logits.argmax(dim=1) == targets).sum().item()  # (B, C) -> (B,)
            total += n
    # Dataset-level means; max(1, total) guards a zero-length loader.
    return total_loss / max(1, total), total_correct / max(1, total)


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    run_dir: str | Path | None = None,
    device: str | torch.device = "cpu",
    label_list: list[str] | None = None,
    progress: bool = False,
) -> dict:
    """Core training loop; with run_dir=None no files are written.

    Logs per-epoch metrics to {run_dir}/metrics.csv and saves the
    best-by-val-accuracy checkpoint to {run_dir}/best.pt when run_dir is set.
    Returns a dict with best_val_acc, best_epoch and the per-epoch history.
    """
    device = torch.device(device)
    model.to(device)
    epochs = int(cfg["epochs"])
    # Label smoothing (default 0.1) spreads a little target mass over the wrong classes so
    # the model cannot chase infinitely confident logits — a cheap regulariser given the
    # small per-word clip counts.
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(cfg.get("label_smoothing", 0.1))
    )
    # AdamW decouples weight decay from the adaptive step, giving true L2-style shrinkage.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )
    scheduler = make_scheduler(optimizer, epochs, int(cfg.get("warmup_epochs", 0)))

    metrics_path = None
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.csv"
        # Write the header once, then append one row per epoch below, so the full learning
        # curve survives on disk even if the run is interrupted mid-training.
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
            )

    best_val_acc = -1.0
    best_epoch = -1
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        # Read the lr before scheduler.step() so the logged value is the one trained with.
        lr_now = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, progress=progress, desc=f"epoch {epoch}",
        )
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device)
        scheduler.step()  # per-epoch schedule: advance the warm-up/cosine factor

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": lr_now,
            }
        )
        if metrics_path is not None:
            with open(metrics_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        epoch,
                        f"{train_loss:.6f}",
                        f"{train_acc:.6f}",
                        f"{val_loss:.6f}",
                        f"{val_acc:.6f}",
                        f"{lr_now:.8e}",
                    ]
                )
        print(
            f"epoch {epoch:3d}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | lr {lr_now:.2e}"
        )

        # Model selection uses val accuracy only (strict '>' keeps the earliest epoch on
        # ties); the held-out test set is never consulted here, preserving the one-shot
        # test discipline. best.pt is overwritten in place, keeping only the best weights.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            if run_dir is not None:
                # Bundle config + label_list so evaluation can rebuild the exact model and
                # class ordering from best.pt alone.
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": cfg,
                        "label_list": label_list,
                        "val_acc": float(val_acc),
                        "epoch": epoch,
                    },
                    Path(run_dir) / "best.pt",
                )

    return {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "final_train_loss": history[-1]["train_loss"] if history else float("nan"),
        "final_val_loss": history[-1]["val_loss"] if history else float("nan"),
        "history": history,
    }


def overfit_one_batch(
    model: nn.Module,
    train_loader: DataLoader,
    cfg: dict,
    device: str | torch.device,
    max_steps: int = 200,
    target_acc: float = 0.95,
) -> dict:
    """Sanity check: fit a single fixed batch; assert >=95% acc within 200 steps."""
    # Any correctly wired model of this capacity should memorise one batch. Failure points
    # to a plumbing bug (wrong shapes, detached graph, frozen weights, mis-scaled lr) —
    # passing says nothing about generalisation, only that learning works end to end.
    device = torch.device(device)
    model.to(device)
    model.train()
    inputs, targets = next(iter(train_loader))  # take one batch and reuse it every step
    inputs = inputs.to(device)
    targets = targets.to(device)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(cfg.get("label_smoothing", 0.1))
    )
    # No weight decay: regularisation works against memorising one batch.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["lr"]), weight_decay=0.0
    )
    acc = 0.0
    step = 0
    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        # Accuracy from this step's forward pass (i.e. weights before the update above).
        acc = (logits.argmax(dim=1) == targets).float().mean().item()
        if step % 20 == 0 or acc >= target_acc:
            print(f"step {step:3d} | loss {loss.item():.4f} | batch acc {acc:.3f}")
        if acc >= target_acc:
            break
    assert acc >= target_acc, (
        f"overfit-batch FAIL: batch acc {acc:.3f} < {target_acc} "
        f"after {max_steps} steps"
    )
    print(f"overfit-batch PASS: batch acc {acc:.3f} within {step} steps")
    return {"overfit_steps": step, "overfit_acc": acc}


def write_env_file(run_dir: Path) -> None:
    """Record python/torch/numpy versions and the git commit hash."""
    # Provenance for the dissertation: every run directory records exactly which code and
    # library versions produced it, so any reported figure can be traced back and re-run.
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip() or "unknown"
    except Exception:
        commit = "unknown"  # not a git checkout / git absent: still record the versions
    lines = [
        f"python: {platform.python_version()}",
        f"torch: {torch.__version__}",
        f"numpy: {np.__version__}",
        f"git_commit: {commit}",
    ]
    (run_dir / "env.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_training(
    cfg: dict,
    device: str | None = None,
    overfit_batch: bool = False,
    shuffle_labels: bool = False,
    progress: bool = True,
) -> dict:
    """Full training run from a config dict; returns {'best_val_acc', 'run_dir', ...}.

    Programmatic equivalent of the CLI so notebooks can call it directly.
    """
    cfg = dict(cfg)  # shallow copy so overrides never mutate the caller's dict
    # Seed before any model or data construction so weight init, shuffles and augmentation
    # are all reproducible from cfg['seed'] alone.
    set_seed(int(cfg["seed"]))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, label_list = build_dataloaders(
        cfg, shuffle_labels=shuffle_labels
    )
    model = build_model(cfg)
    print(
        f"[train] built {cfg['arch']} model | "
        f"trainable parameters: {count_parameters(model):,}"
    )
    print(
        f"[train] device={device} | train clips={len(train_loader.dataset)} | "
        f"val clips={len(val_loader.dataset)} | augment={cfg.get('augment', False)}"
    )

    if overfit_batch:
        # Sanity mode writes nothing to disk; NaN val acc marks it as a check, not a result.
        result = overfit_one_batch(model, train_loader, cfg, device)
        return {"best_val_acc": float("nan"), "run_dir": None, **result}

    # Run ids name the grid cell, e.g. 'transformer_aug_s42' — one directory per run of
    # the 12-run experiment grid ({arch} x {aug,noaug} x seeds 42/43/44).
    run_id = cfg.get("run_id") or (
        f"{cfg['arch']}_{'aug' if cfg.get('augment') else 'noaug'}_s{cfg['seed']}"
    )
    if shuffle_labels:
        run_id = f"{run_id}_shufflelabels"  # never clobber a real run
    run_dir = Path(cfg.get("out_dir", "results/runs")) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # Snapshot the exact (post-override) config plus environment/commit provenance next to
    # the results, so every run directory is self-describing.
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    write_env_file(run_dir)

    result = train_loop(
        model,
        train_loader,
        val_loader,
        cfg,
        run_dir=run_dir,
        device=device,
        label_list=label_list,
        progress=progress,
    )
    print(
        f"[train] done: best val acc {result['best_val_acc']:.4f} "
        f"at epoch {result['best_epoch']} | run dir: {run_dir}"
    )
    return {
        "best_val_acc": result["best_val_acc"],
        "run_dir": str(run_dir),
        "best_epoch": result["best_epoch"],
        "history": result["history"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a BSL word classifier.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument(
        "--overfit-batch",
        action="store_true",
        help="Sanity mode: fit one fixed batch, assert >=95%% acc in 200 steps.",
    )
    parser.add_argument(
        "--shuffle-labels",
        action="store_true",
        help="Sanity mode: permute training labels once; val acc ~ chance.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Default: cuda if available else cpu.",
    )
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs  # override is recorded in the saved config.yaml snapshot

    run_training(
        cfg,
        device=args.device,
        overfit_batch=args.overfit_batch,
        shuffle_labels=args.shuffle_labels,
    )


if __name__ == "__main__":
    main()
