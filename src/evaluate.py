"""Evaluation and statistical comparison of trained BSL classifiers.

Usage:
    python -m src.evaluate --checkpoint results/runs/x/best.pt --split val
    python -m src.evaluate --checkpoint results/runs/x/best.pt --split test
    python -m src.evaluate --mcnemar runA/eval_val/predictions.csv runB/eval_val/predictions.csv

Writes, per evaluation run, into {run_dir}/eval_{split}/:
    predictions.csv, per_class_accuracy.csv, confusion_matrix.csv, confusion_matrix.png
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.metrics import confusion_matrix, f1_score

ALPHA = 0.05


def word_from_clip_id(clip_id: str) -> str:
    """Recover the label word from a clip id of the form {word}_{source}_{nnn}."""
    # rsplit from the right (2 splits) so underscores inside the word itself survive intact.
    return clip_id.rsplit("_", 2)[0]


def compute_topk_accuracy(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Fraction of samples whose true label is among the k largest logits."""
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    # Stable sort so tie-breaking (and therefore the metric) is deterministic.
    # Negate to sort descending (argsort has no reverse flag); (N, C) -> (N, k) class indices.
    topk = np.argsort(-logits, axis=1, kind="stable")[:, :k]
    # labels[:, None] is (N, 1): broadcasts against (N, k) so each sample checks its k guesses.
    return float((topk == labels[:, None]).any(axis=1).mean())


def compute_macro_f1(logits: np.ndarray, labels: np.ndarray) -> float:
    """Macro-F1 over ALL model classes (zero support classes score 0)."""
    logits = np.asarray(logits)
    preds = logits.argmax(axis=1)
    # Passing labels=all_classes averages over every class the model can output, not only the
    # classes present in this split; zero_division=0 scores never-predicted/zero-support classes
    # as 0 instead of warning. Keeps macro-F1 comparable between val and the smaller test set.
    all_classes = list(range(logits.shape[1]))
    return float(
        f1_score(labels, preds, average="macro", labels=all_classes, zero_division=0)
    )


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact-binomial and continuity-corrected chi-square McNemar's test.

    Args are paired boolean/0-1 arrays of per-sample correctness. Returns a
    dict with discordant counts b (A right, B wrong), c (A wrong, B right),
    the exact two-sided binomial p-value and the chi-square statistic/p-value.
    """
    a = np.asarray(correct_a).astype(bool)
    b_arr = np.asarray(correct_b).astype(bool)
    if a.shape != b_arr.shape:
        raise ValueError(f"Paired arrays differ in shape: {a.shape} vs {b_arr.shape}")
    # McNemar's test uses only the discordant pairs (clips the two models disagree on); pairs
    # where both are right or both wrong carry no information about which model is better.
    b = int(np.sum(a & ~b_arr))  # A right, B wrong
    c = int(np.sum(~a & b_arr))  # A wrong, B right
    n = b + c
    if n == 0:
        # Models agree on every clip: no evidence of a difference, so p = 1 by convention.
        return {"b": 0, "c": 0, "n_discordant": 0, "exact_p": 1.0,
                "chi2_stat": 0.0, "chi2_p": 1.0}
    # Under H0 the models are equally good, so each discordant pair is a fair coin flip:
    # b ~ Binomial(n, 0.5). The exact two-sided binomial p-value stays valid for the small
    # discordant counts expected here, where the chi-square approximation is least reliable.
    exact_p = float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
    # Edwards continuity correction (the -1) compensates for approximating discrete counts
    # with the continuous chi-square distribution; reported alongside the exact test.
    chi2_stat = (abs(b - c) - 1) ** 2 / n
    chi2_p = float(chi2.sf(chi2_stat, df=1))  # sf = 1 - CDF: upper-tail p, chi-square with 1 df
    return {"b": b, "c": c, "n_discordant": n, "exact_p": exact_p,
            "chi2_stat": float(chi2_stat), "chi2_p": chi2_p}


def align_predictions(
    df_a: pd.DataFrame, df_b: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Inner-join two predictions frames on clip_id; return paired 'correct' arrays."""
    # Joining on clip_id (rather than trusting row order) guarantees a genuinely paired sample:
    # each row compares both models on exactly the same clip, as McNemar's test requires. The
    # inner join silently drops clips only one run saw.
    merged = df_a[["clip_id", "correct"]].merge(
        df_b[["clip_id", "correct"]], on="clip_id", suffixes=("_a", "_b")
    )
    if merged.empty:
        raise ValueError("No overlapping clip_ids between the two prediction files")
    return (
        merged["correct_a"].to_numpy().astype(bool),
        merged["correct_b"].to_numpy().astype(bool),
    )


def _collect_logits(model, loader) -> tuple[np.ndarray, np.ndarray]:
    import torch

    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    # inference_mode disables autograd tracking entirely (cheaper than no_grad); eval only.
    with torch.inference_mode():
        for features, labels in loader:
            all_logits.append(model(features).numpy())  # (B, 64, 315) features -> (B, C) logits
            all_labels.append(np.asarray(labels))
    # Stack per-batch chunks into one (N, C) logits array and one (N,) label array.
    return np.concatenate(all_logits), np.concatenate(all_labels)


def _plot_confusion_matrix(
    cm: np.ndarray, label_list: list[str], out_png: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless backend: render straight to file, no display required
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="viridis", interpolation="nearest")
    ticks = range(len(label_list))
    ax.set_xticks(ticks, labels=label_list, rotation=90, fontsize=6)
    ax.set_yticks(ticks, labels=label_list, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="count")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _gather_split(
    split: str, cfg: dict, label_list: list[str], landmarks_dir: str | None
) -> tuple[list[str], list[int], str]:
    """Return (clip_ids, labels, landmarks_dir) for the requested split."""
    label_index = {w: i for i, w in enumerate(label_list)}  # word -> integer class id
    if split == "val":
        from src.dataset import load_splits

        # Validation clips come from the frozen splits.json written at split time; that split
        # is grouped by source organisation, so no training signer can leak into evaluation.
        splits = load_splits(cfg.get("splits_file", "data/splits.json"))
        lm_dir = landmarks_dir or cfg.get("landmarks_dir", "data/landmarks")
        clip_ids = list(splits["val"])
        words = [word_from_clip_id(c) for c in clip_ids]  # label is encoded in the clip id
    else:
        # The self-recorded held-out test set lives in one folder per word (the folder name is
        # the ground-truth label); by protocol it is evaluated exactly once, after model choice.
        lm_dir = landmarks_dir or "data/test_landmarks"
        clip_ids, words = [], []
        root = Path(lm_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"Test landmarks directory not found: {root}")
        # sorted() makes discovery order (hence CSV row order) deterministic across machines.
        for word_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            # Words the checkpoint was never trained on are skipped with a warning, not an error,
            # so a superset test recording can still be scored on the model's 50-word vocabulary.
            if word_dir.name not in label_index:
                warnings.warn(
                    f"Skipping folder {word_dir.name!r}: not in checkpoint label_list"
                )
                continue
            for npz in sorted(word_dir.glob("*.npz")):
                clip_ids.append(npz.stem)
                words.append(word_dir.name)
    # Fail fast if any gathered word cannot be mapped to a class id (e.g. a malformed clip id):
    # silently mislabelled evaluation data would invalidate every downstream metric.
    missing = sorted({w for w in words if w not in label_index})
    if missing:
        raise ValueError(f"Words not in checkpoint label_list: {missing}")
    return clip_ids, [label_index[w] for w in words], lm_dir


def run_evaluation(
    checkpoint: str,
    split: str,
    landmarks_dir: str | None = None,
    batch_size: int = 32,
) -> dict:
    """Evaluate a checkpoint on a split; write CSV/PNG artefacts; return metrics."""
    import torch
    from torch.utils.data import DataLoader

    from src.dataset import BSLDataset
    from src.models import build_model

    # The checkpoint bundles weights with the training config and label list, so evaluation
    # rebuilds exactly the trained architecture and index->word mapping. weights_only=False is
    # needed because config/label_list are plain Python objects; CPU matches the laptop target.
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg: dict = ckpt["config"]
    label_list: list[str] = list(ckpt["label_list"])
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()  # switches dropout (and any batch-norm) to deterministic inference behaviour

    clip_ids, labels, lm_dir = _gather_split(split, cfg, label_list, landmarks_dir)
    if not clip_ids:
        raise ValueError(f"No clips found for split {split!r} in {lm_dir}")

    # No augmentation and no shuffling: evaluation must be deterministic, and preserving the
    # dataset order keeps the logits rows aligned with clip_ids for the predictions CSV.
    dataset = BSLDataset(
        clip_ids, labels, lm_dir, metadata=None, augment=False,
        seq_len=cfg.get("seq_len", 64),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    logits, labels_arr = _collect_logits(model, loader)
    preds = logits.argmax(axis=1)
    correct = (preds == labels_arr).astype(int)

    n_classes = logits.shape[1]
    top1 = compute_topk_accuracy(logits, labels_arr, 1)
    top5 = compute_topk_accuracy(logits, labels_arr, 5)
    macro_f1 = compute_macro_f1(logits, labels_arr)

    out_dir = Path(checkpoint).resolve().parent / f"eval_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    k = min(5, n_classes)  # guard for tiny debug vocabularies with fewer than 5 classes
    # Same stable descending sort as compute_topk_accuracy: (N, C) logits -> (N, k) ranked ids.
    top5_idx = np.argsort(-logits, axis=1, kind="stable")[:, :k]
    pred_rows: dict = {
        "clip_id": clip_ids,
        "true": [label_list[i] for i in labels_arr],
        "pred": [label_list[i] for i in preds],
        "correct": correct,
    }
    for j in range(5):
        # Always emit five top5_* columns for a stable CSV schema, padding with "" when k < 5.
        col = [label_list[i] for i in top5_idx[:, j]] if j < k else [""] * len(clip_ids)
        pred_rows[f"top5_{j + 1}"] = col
    pd.DataFrame(pred_rows).to_csv(out_dir / "predictions.csv", index=False)

    # labels=range(n_classes) forces a full n x n matrix (rows = true, cols = predicted) even
    # when some classes never appear in this split, keeping it aligned with label_list.
    cm = confusion_matrix(labels_arr, preds, labels=list(range(n_classes)))
    support = cm.sum(axis=1)  # row sums: number of true clips per class
    diag = np.diag(cm)  # diagonal: clips of each class that were predicted correctly
    # Per-class accuracy = diagonal / support; 'where' skips zero-support classes so they stay
    # NaN, distinguishing 'no evaluation data' from a genuine 0% accuracy.
    per_class_acc = np.divide(
        diag, support, out=np.full(n_classes, np.nan), where=support > 0
    )
    pd.DataFrame(
        {"word": label_list, "n": support, "correct": diag, "accuracy": per_class_acc}
    ).to_csv(out_dir / "per_class_accuracy.csv", index=False)
    pd.DataFrame(cm, index=label_list, columns=label_list).to_csv(
        out_dir / "confusion_matrix.csv", index_label="true"
    )
    _plot_confusion_matrix(cm, label_list, out_dir / "confusion_matrix.png")

    print(f"Evaluated {len(clip_ids)} clips from split '{split}' ({lm_dir})")
    print(f"  top-1 accuracy : {top1:.4f}")
    print(f"  top-5 accuracy : {top5:.4f}")
    print(f"  macro-F1       : {macro_f1:.4f}")
    print(f"Outputs written to {out_dir}")
    return {
        "n": len(clip_ids), "top1": top1, "top5": top5, "macro_f1": macro_f1,
        "out_dir": str(out_dir),
    }


def run_mcnemar(preds_a: str, preds_b: str) -> dict:
    """McNemar's test on two predictions.csv files (aligned on clip_id)."""
    df_a = pd.read_csv(preds_a)
    df_b = pd.read_csv(preds_b)
    correct_a, correct_b = align_predictions(df_a, df_b)
    res = mcnemar_test(correct_a, correct_b)

    print("McNemar's test on paired predictions")
    print(f"  A: {preds_a}  (accuracy {correct_a.mean():.4f})")
    print(f"  B: {preds_b}  (accuracy {correct_b.mean():.4f})")
    print(f"  aligned pairs           : {len(correct_a)}")
    print(f"  b (A right, B wrong)    : {res['b']}")
    print(f"  c (A wrong, B right)    : {res['c']}")
    print(f"  exact binomial p-value  : {res['exact_p']:.6g}")
    print(
        f"  chi2 (continuity corr.) : {res['chi2_stat']:.4f}, "
        f"p = {res['chi2_p']:.6g}"
    )
    # The exact binomial test decides the verdict: discordant counts on a small test set are
    # exactly where the chi-square approximation (reported above for reference) is weakest.
    verdict = (
        "SIGNIFICANT difference" if res["exact_p"] < ALPHA
        else "NO significant difference"
    )
    print(f"Verdict: {verdict} at alpha={ALPHA} (exact test)")
    return res


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="Path to best.pt")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument(
        "--landmarks-dir", default=None,
        help="Override the landmarks directory for the chosen split",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--mcnemar", nargs=2, metavar=("PREDS_A", "PREDS_B"),
        help="Run McNemar's test on two predictions.csv files instead of evaluating",
    )
    args = parser.parse_args(argv)

    if args.mcnemar:
        # Comparison mode works purely from saved predictions.csv files: no checkpoint, data
        # or model forward passes are needed, so runs can be compared long after training.
        run_mcnemar(args.mcnemar[0], args.mcnemar[1])
        return
    if not args.checkpoint:
        parser.error("--checkpoint is required unless --mcnemar is used")
    run_evaluation(args.checkpoint, args.split, args.landmarks_dir, args.batch_size)


if __name__ == "__main__":
    main()
