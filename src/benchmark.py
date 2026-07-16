"""Latency benchmark for the real-time BSL recognition pipeline.

Usage:
    python -m src.benchmark --checkpoint best.pt --synthetic --frames 1000 --stride 8
    python -m src.benchmark --checkpoint best.pt --video path/to/clip.mp4
    python -m src.benchmark --checkpoint best.pt --webcam 0

Times each stage (capture, holistic, normalise, forward) with
time.perf_counter, reports medians and 95th percentiles over --frames frames
after --warmup warm-up frames, and checks the latency criterion:
>= 15 fps end-to-end and < 100 ms per-window classification.
Appends one row per run to results/latency.csv. --synthetic replaces
capture+holistic with random raw landmark frames (no camera or mediapipe
needed, e.g. for CI).
"""
from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from src.landmarks import N_COORDS, N_LANDMARKS
from src.realtime import holistic_to_landmarks, load_checkpoint, normalise_frame

FPS_TARGET = 15.0
WINDOW_LATENCY_TARGET_MS = 100.0
DEFAULT_OUT_CSV = "results/latency.csv"

CSV_FIELDS = [
    "timestamp", "checkpoint", "mode", "frames", "stride", "seq_len",
    "capture_median_ms", "capture_p95_ms",
    "holistic_median_ms", "holistic_p95_ms",
    "normalise_median_ms", "normalise_p95_ms",
    "forward_median_ms", "forward_p95_ms", "n_windows",
    "end_to_end_fps", "fps_pass", "latency_pass",
    "platform", "processor", "python_version", "torch_version",
]


def _stats(samples: list[float]) -> tuple[float, float]:
    """(median, p95) of a list of per-stage timings in ms."""
    arr = np.asarray(samples)
    return float(np.median(arr)), float(np.percentile(arr, 95))


def _open_capture(mode: str, source: str | int):
    import cv2

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {mode} source: {source!r}")
    return cap


def _create_holistic():
    try:
        import mediapipe as mp
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "mediapipe is required for --video/--webcam benchmarking; "
            "use --synthetic to benchmark without it"
        ) from exc
    return mp.solutions.holistic.Holistic(model_complexity=1)


def run_benchmark(
    checkpoint: str,
    mode: str,
    source: str | int | None = None,
    frames: int = 1000,
    stride: int = 8,
    warmup: int = 32,
    out_csv: str = DEFAULT_OUT_CSV,
) -> dict:
    """Run the benchmark; print the report; append a row to out_csv.

    mode is one of 'synthetic', 'video', 'webcam'. Returns the results dict
    (also written to out_csv), including boolean pass flags.
    """
    model, cfg, _ = load_checkpoint(checkpoint)
    seq_len = cfg.get("seq_len", 64)
    in_dim = cfg.get("in_dim", 315)

    with torch.inference_mode():  # model warm-up
        for _ in range(3):
            model(torch.zeros(1, seq_len, in_dim))

    cap = None
    holistic = None
    if mode != "synthetic":
        import cv2

        cap = _open_capture(mode, source)
        holistic = _create_holistic()

    rng = np.random.default_rng(42)
    buffer: deque[np.ndarray] = deque(maxlen=seq_len)
    capture_ms: list[float] = []
    holistic_ms: list[float] = []
    normalise_ms: list[float] = []
    forward_ms: list[float] = []
    measured = 0
    wall_start: float | None = None

    for i in range(frames + warmup):
        measuring = i >= warmup
        if measuring and wall_start is None:
            wall_start = time.perf_counter()

        if mode == "synthetic":
            raw = rng.random((N_LANDMARKS, N_COORDS)).astype(np.float32)
        else:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok and mode == "video":  # loop the clip to reach --frames
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            if not ok:
                print(f"Source exhausted after {i} frames; stopping early.")
                break
            t1 = time.perf_counter()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            raw = holistic_to_landmarks(results)
            t2 = time.perf_counter()
            if measuring:
                capture_ms.append((t1 - t0) * 1e3)
                holistic_ms.append((t2 - t1) * 1e3)

        t3 = time.perf_counter()
        buffer.append(normalise_frame(raw))
        t4 = time.perf_counter()
        if measuring:
            normalise_ms.append((t4 - t3) * 1e3)

        if len(buffer) == seq_len and i % stride == 0:
            t5 = time.perf_counter()
            window = np.stack(buffer).reshape(1, seq_len, -1)
            with torch.inference_mode():
                torch.softmax(model(torch.from_numpy(window)), dim=-1)
            t6 = time.perf_counter()
            if measuring:
                forward_ms.append((t6 - t5) * 1e3)

        if measuring:
            measured += 1

    wall = time.perf_counter() - (wall_start if wall_start is not None else 0.0)
    if cap is not None:
        cap.release()
    if holistic is not None:
        holistic.close()
    if measured == 0:
        raise RuntimeError("No frames measured; increase --frames or check source")

    end_to_end_fps = measured / wall
    fps_pass = end_to_end_fps >= FPS_TARGET
    latency_pass = False
    fwd_med = fwd_p95 = None
    if forward_ms:
        fwd_med, fwd_p95 = _stats(forward_ms)
        latency_pass = fwd_med < WINDOW_LATENCY_TARGET_MS

    print(f"Benchmark: mode={mode} frames={measured} stride={stride} "
          f"seq_len={seq_len} (warmup {warmup} frames excluded)")
    print("Stage timings (ms):")
    for name, samples in (
        ("capture", capture_ms), ("holistic", holistic_ms),
        ("normalise", normalise_ms), ("forward", forward_ms),
    ):
        if samples:
            med, p95 = _stats(samples)
            print(f"  {name:<10} median {med:8.3f}   p95 {p95:8.3f}   "
                  f"(n={len(samples)})")
        else:
            print(f"  {name:<10} skipped ({mode} mode)")
    print(f"End-to-end: {measured} frames in {wall:.2f} s -> "
          f"{end_to_end_fps:.1f} fps"
          + (" (synthetic: capture+holistic skipped)"
             if mode == "synthetic" else ""))
    if fwd_med is not None:
        print(f"Per-window classification: median {fwd_med:.2f} ms, "
              f"p95 {fwd_p95:.2f} ms over {len(forward_ms)} windows")
    else:
        print("Per-window classification: no windows classified "
              f"(need >= {seq_len} frames)")

    print(f"{'PASS' if fps_pass else 'FAIL'}: end-to-end throughput "
          f"{end_to_end_fps:.1f} fps vs target >= {FPS_TARGET:.0f} fps")
    if fwd_med is not None:
        print(f"{'PASS' if latency_pass else 'FAIL'}: per-window latency "
              f"median {fwd_med:.2f} ms vs target < "
              f"{WINDOW_LATENCY_TARGET_MS:.0f} ms")
    else:
        print("FAIL: per-window latency could not be measured")

    def fmt(samples: list[float], which: int) -> str:
        return f"{_stats(samples)[which]:.3f}" if samples else ""

    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": checkpoint,
        "mode": mode,
        "frames": measured,
        "stride": stride,
        "seq_len": seq_len,
        "capture_median_ms": fmt(capture_ms, 0),
        "capture_p95_ms": fmt(capture_ms, 1),
        "holistic_median_ms": fmt(holistic_ms, 0),
        "holistic_p95_ms": fmt(holistic_ms, 1),
        "normalise_median_ms": fmt(normalise_ms, 0),
        "normalise_p95_ms": fmt(normalise_ms, 1),
        "forward_median_ms": fmt(forward_ms, 0),
        "forward_p95_ms": fmt(forward_ms, 1),
        "n_windows": len(forward_ms),
        "end_to_end_fps": f"{end_to_end_fps:.2f}",
        "fps_pass": fps_pass,
        "latency_pass": latency_pass,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"Row appended to {out_path}")
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--video", help="Benchmark against a video file")
    group.add_argument("--webcam", type=int, help="Benchmark against a webcam index")
    group.add_argument("--synthetic", action="store_true",
                       help="Random landmark frames; no camera or mediapipe needed")
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--out", default=DEFAULT_OUT_CSV)
    args = parser.parse_args(argv)

    if args.video is not None:
        mode, source = "video", args.video
    elif args.webcam is not None:
        mode, source = "webcam", args.webcam
    else:
        if not args.synthetic:
            print("No source given; defaulting to --synthetic.")
        mode, source = "synthetic", None

    row = run_benchmark(
        args.checkpoint, mode, source,
        frames=args.frames, stride=args.stride, warmup=args.warmup,
        out_csv=args.out,
    )
    return 0 if (row["fps_pass"] and row["latency_pass"]) else 1


if __name__ == "__main__":
    sys.exit(main())
