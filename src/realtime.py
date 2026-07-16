"""Real-time webcam BSL recognition demo.

Usage:
    python -m src.realtime --checkpoint results/runs/x/best.pt [--camera 0]
        [--stride 8] [--threshold 0.6] [--consecutive 3] [--no-mirror]

Keeps a rolling buffer of the last 64 normalised frames, classifies every
--stride frames, and displays a prediction only once the same top-1 class has
cleared the confidence threshold for --consecutive consecutive windows.
Press q to quit.
"""
from __future__ import annotations

import argparse
import time
from collections import deque

import numpy as np
import torch
from torch import nn

try:
    import cv2
except ImportError:  # deferred: cv2 only needed when the live demo runs
    cv2 = None

from src.landmarks import (
    LEFT_HAND_SLICE,
    MOUTH_FACE_INDICES,
    MOUTH_SLICE,
    N_COORDS,
    N_LANDMARKS,
    POSE_SLICE,
    RIGHT_HAND_SLICE,
)
from src.models import build_model
from src.normalise import normalise_sequence


def load_checkpoint(path: str) -> tuple[nn.Module, dict, list[str]]:
    """Load best.pt; return (model in eval mode, config dict, label_list)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["config"], list(ckpt["label_list"])


def holistic_to_landmarks(results) -> np.ndarray:
    """Convert one MediaPipe Holistic result to a (105, 3) array, NaN = missing."""
    arr = np.full((N_LANDMARKS, N_COORDS), np.nan, dtype=np.float32)

    def fill(block: slice, landmark_list) -> None:
        for i, lm in enumerate(landmark_list.landmark):
            arr[block.start + i] = (lm.x, lm.y, lm.z)

    if results.pose_landmarks is not None:
        fill(POSE_SLICE, results.pose_landmarks)
    if results.left_hand_landmarks is not None:
        fill(LEFT_HAND_SLICE, results.left_hand_landmarks)
    if results.right_hand_landmarks is not None:
        fill(RIGHT_HAND_SLICE, results.right_hand_landmarks)
    if results.face_landmarks is not None:
        face = results.face_landmarks.landmark
        for j, idx in enumerate(MOUTH_FACE_INDICES):
            arr[MOUTH_SLICE.start + j] = (face[idx].x, face[idx].y, face[idx].z)
    return arr


def normalise_frame(raw: np.ndarray) -> np.ndarray:
    """Normalise a single raw (105, 3) frame; missing landmarks become 0."""
    return np.nan_to_num(normalise_sequence(raw[None])[0], nan=0.0).astype(np.float32)


class PredictionSmoother:
    """Confirm a class once its top-1 confidence clears `threshold` for
    `consecutive` consecutive windows; hold the last confirmed prediction."""

    def __init__(self, threshold: float = 0.6, consecutive: int = 3) -> None:
        self.threshold = threshold
        self.consecutive = consecutive
        self._candidate: int | None = None
        self._streak = 0
        self._confirmed: int | None = None
        self._confirmed_conf = 0.0

    def update(self, class_idx: int, confidence: float) -> tuple[int | None, float]:
        """Feed one window's top-1; return (confirmed class or None, its conf)."""
        if confidence >= self.threshold:
            if class_idx == self._candidate:
                self._streak += 1
            else:
                self._candidate = class_idx
                self._streak = 1
        else:
            self._candidate = None
            self._streak = 0
        if self._streak >= self.consecutive:
            self._confirmed = class_idx
            self._confirmed_conf = confidence
        return self._confirmed, self._confirmed_conf


def _import_mediapipe():
    try:
        import mediapipe as mp
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "mediapipe is required for real-time inference; "
            "install it with: pip install mediapipe==0.10.9"
        ) from exc
    return mp


def run(
    checkpoint: str,
    camera: int = 0,
    stride: int = 8,
    threshold: float = 0.6,
    consecutive: int = 3,
    mirror: bool = True,
) -> None:
    """Open the webcam and run the live recognition loop until q is pressed."""
    if cv2 is None:
        raise RuntimeError(
            "opencv-python is required for the real-time demo; "
            "install it with: pip install opencv-python"
        )
    mp = _import_mediapipe()
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    model, cfg, label_list = load_checkpoint(checkpoint)
    seq_len = cfg.get("seq_len", 64)
    smoother = PredictionSmoother(threshold=threshold, consecutive=consecutive)
    buffer: deque[np.ndarray] = deque(maxlen=seq_len)

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera}")

    holistic = mp_holistic.Holistic(model_complexity=1)
    window_name = "BSL Recognition (q to quit)"
    frame_idx = 0
    fps = 0.0
    last_t = time.perf_counter()
    latest: tuple[str, float] | None = None  # most recent (unsmoothed) window top-1
    confirmed_idx: int | None = None
    confirmed_conf = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame grab failed; exiting.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            buffer.append(normalise_frame(holistic_to_landmarks(results)))

            if len(buffer) == seq_len and frame_idx % stride == 0:
                window = np.stack(buffer).reshape(1, seq_len, -1)
                with torch.inference_mode():
                    probs = torch.softmax(model(torch.from_numpy(window)), dim=-1)
                conf, cls = torch.max(probs[0], dim=0)
                latest = (label_list[int(cls)], float(conf))
                confirmed_idx, confirmed_conf = smoother.update(
                    int(cls), float(conf)
                )

            # Draw skeleton on the unmirrored frame so overlays flip with it.
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
            )
            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
            )
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
            )
            if mirror:
                frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            inst = 1.0 / max(now - last_t, 1e-6)
            fps = inst if fps == 0.0 else 0.9 * fps + 0.1 * inst
            last_t = now

            hud = frame.copy()
            cv2.rectangle(hud, (0, 0), (frame.shape[1], 110), (0, 0, 0), -1)
            frame = cv2.addWeighted(hud, 0.45, frame, 0.55, 0)
            if confirmed_idx is not None:
                text = f"{label_list[confirmed_idx]}  {confirmed_conf:.2f}"
                colour = (0, 255, 0)
            else:
                text = "listening..."
                colour = (200, 200, 200)
            cv2.putText(frame, text, (10, 42), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, colour, 2, cv2.LINE_AA)
            window_text = (
                f"window: {latest[0]} {latest[1]:.2f}" if latest else "window: -"
            )
            cv2.putText(frame, window_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"{fps:.1f} fps | buffer {len(buffer)}/{seq_len}",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_idx += 1
    finally:
        holistic.close()
        cap.release()
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--stride", type=int, default=8,
                        help="Classify every k-th frame")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--consecutive", type=int, default=3)
    parser.add_argument("--mirror", action=argparse.BooleanOptionalAction,
                        default=True, help="Mirror the display (selfie view)")
    args = parser.parse_args(argv)
    run(
        args.checkpoint,
        camera=args.camera,
        stride=args.stride,
        threshold=args.threshold,
        consecutive=args.consecutive,
        mirror=args.mirror,
    )


if __name__ == "__main__":
    main()
