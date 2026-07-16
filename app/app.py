"""Gradio demo for isolated word-level BSL recognition (Mode A).

Upload or webcam-record a short clip of a single BSL sign; the app runs
MediaPipe Holistic landmark extraction, the shared preprocessing pipeline
and a trained classifier, then shows the top-5 predicted signs.

Research demonstrator for BSL learning and lookup -- not an interpreting
service.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
# Support both layouts: src/ next to app.py (HF Spaces) or in the repo root.
for _path in (REPO_ROOT, APP_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import gradio as gr  # noqa: E402

from src.dataset import resample_sequence  # noqa: E402
from src.extract import extract_video, fill_gaps  # noqa: E402
from src.landmarks import FEATURES_PER_FRAME  # noqa: E402
from src.models import build_model  # noqa: E402
from src.normalise import normalise_sequence  # noqa: E402

SEQ_LEN = 64
MIN_FRAMES = 8

HEADER_MD = """
# BSL Word Recognition

**Research demonstrator for BSL learning and lookup — not an interpreting
service.** This app recognises isolated signs from a fixed 50-word British
Sign Language vocabulary using a pose-based transformer. It is an MSc
research prototype: predictions can be wrong, and it must not be relied on
for communication, interpreting, or any safety-critical purpose.

**How to use**

1. Upload — or record with your webcam — a 2–3 second clip of a single BSL
   sign. One signer, facing the camera, upper body and both hands clearly
   visible.
2. Click **Recognise**.
3. The five most likely signs and their confidences appear on the right.
"""

NO_CHECKPOINT_MSG = (
    "No trained checkpoint found. Set the BSL_CHECKPOINT environment "
    "variable to a checkpoint path, copy one to app/model/best.pt, or train "
    "a model so that results/runs/<run_id>/best.pt exists."
)

_MODEL: torch.nn.Module | None = None
_LABELS: list[str] | None = None
_CKPT_PATH: Path | None = None
_SEQ_LEN: int = SEQ_LEN


def find_checkpoint() -> Path | None:
    """Resolve a checkpoint: env var, then app/model/best.pt, then newest run."""
    env_path = os.environ.get("BSL_CHECKPOINT")
    if env_path and Path(env_path).is_file():
        return Path(env_path)
    bundled = APP_DIR / "model" / "best.pt"
    if bundled.is_file():
        return bundled
    runs = sorted(
        (REPO_ROOT / "results" / "runs").glob("*/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return runs[0] if runs else None


def load_model() -> tuple[torch.nn.Module, list[str], Path]:
    """Load the classifier once and cache it."""
    global _MODEL, _LABELS, _CKPT_PATH, _SEQ_LEN
    if _MODEL is None:
        path = find_checkpoint()
        if path is None:
            raise FileNotFoundError(NO_CHECKPOINT_MSG)
        ckpt = torch.load(path, map_location="cpu")
        model = build_model(ckpt["config"])
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        _MODEL, _LABELS, _CKPT_PATH = model, list(ckpt["label_list"]), path
        _SEQ_LEN = int(ckpt["config"].get("seq_len", SEQ_LEN))
    assert _LABELS is not None and _CKPT_PATH is not None
    return _MODEL, _LABELS, _CKPT_PATH


def make_holistic():
    """Fresh MediaPipe Holistic per clip: clips are independent, so tracking
    state must not carry over between requests (matches src/extract.py)."""
    import mediapipe as mp

    return mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
    )


@torch.no_grad()
def recognise(video_path: str | None) -> tuple[dict[str, float], str]:
    """Classify one clip; returns (label confidences, status markdown)."""
    if not video_path:
        raise gr.Error("Please upload or record a short clip first.")
    try:
        model, labels, ckpt_path = load_model()
    except FileNotFoundError as exc:
        raise gr.Error(str(exc)) from exc

    try:
        with make_holistic() as holistic:
            landmarks, presence, fps = extract_video(video_path, holistic)
    except Exception as exc:  # unreadable file, codec issues, etc.
        raise gr.Error(f"Could not process the video: {exc}") from exc

    n_frames = int(landmarks.shape[0])
    if n_frames < MIN_FRAMES:
        raise gr.Error(
            f"Only {n_frames} frames could be read from the clip. "
            "Please record 2-3 seconds of video."
        )

    seq = fill_gaps(landmarks, presence)
    seq = normalise_sequence(seq)
    seq = np.nan_to_num(seq, nan=0.0)
    seq = resample_sequence(seq, _SEQ_LEN).astype(np.float32)

    features = torch.from_numpy(
        np.ascontiguousarray(seq.reshape(1, _SEQ_LEN, FEATURES_PER_FRAME))
    )
    probs = torch.softmax(model(features), dim=-1)[0]
    confidences = {label: float(p) for label, p in zip(labels, probs)}

    left_pct = 100.0 * float(presence[:, 0].mean())
    right_pct = 100.0 * float(presence[:, 1].mean())
    any_hand_pct = 100.0 * float(presence[:, :2].max(axis=1).mean())
    face_pct = 100.0 * float(presence[:, 2].mean())
    status = (
        f"Processed **{n_frames}** frames at {fps:.1f} fps. "
        f"Hands detected in **{any_hand_pct:.0f}%** of frames "
        f"(left {left_pct:.0f}%, right {right_pct:.0f}%); "
        f"face in {face_pct:.0f}%. "
        f"Checkpoint: `{ckpt_path.name}` "
        f"({ckpt_path.parent.name})."
    )
    if any_hand_pct < 50.0:
        status += (
            "\n\n*Hands were hard to detect — try better lighting and keep "
            "both hands inside the frame.*"
        )
    return confidences, status


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="BSL Word Recognition") as demo:
        gr.Markdown(HEADER_MD)
        if find_checkpoint() is None:
            gr.Markdown(f"**Warning:** {NO_CHECKPOINT_MSG}")
        with gr.Row():
            with gr.Column():
                video = gr.Video(
                    sources=["upload", "webcam"],
                    label="Sign clip (2-3 seconds)",
                    include_audio=False,
                )
                button = gr.Button("Recognise", variant="primary")
            with gr.Column():
                label = gr.Label(num_top_classes=5, label="Top-5 predictions")
                status = gr.Markdown()
        button.click(recognise, inputs=video, outputs=[label, status])
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.queue(max_size=8).launch()
