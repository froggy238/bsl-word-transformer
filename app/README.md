---
title: BSL Word Recognition
emoji: "\U0001F450"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# BSL Word Recognition — Research Demonstrator

**Research demonstrator for BSL learning and lookup — not an interpreting
service.**

This Space is the demo for an MSc project, *Real-Time British Sign Language
Recognition: A Pose-Based Transformer Approach for Word-Level BSL
Classification*. It recognises isolated signs from a fixed 50-word BSL
vocabulary: a short clip is converted to MediaPipe Holistic pose/hand/mouth
landmarks, normalised, resampled to 64 frames, and classified by a
from-scratch transformer encoder (~1M parameters).

It is a research prototype intended for sign learning and dictionary-style
lookup. Predictions can be wrong, the vocabulary is limited to 50 words, and
it must not be used as a substitute for qualified BSL interpreters or for any
communication-critical purpose.

Source code: [GitHub repository](https://github.com/froggy238/bsl-word-transformer)

## Usage

1. Upload — or record with your webcam — a 2–3 second clip of a single BSL
   sign (one signer facing the camera, upper body and both hands visible).
2. Click **Recognise**.
3. The top-5 predicted signs are shown with confidences, along with landmark
   detection statistics for the clip.

## Deploying this Space

The app expects the shared `src/` package and a trained checkpoint next to
`app.py`. From the project repository root:

1. Create a new Gradio Space (CPU hardware is sufficient).
2. Copy into the Space repository root:
   - `app/app.py` as `app.py`
   - `app/requirements.txt` as `requirements.txt`
   - `app/README.md` as `README.md`
   - the whole `src/` directory as `src/`
3. Copy your best trained checkpoint (e.g.
   `results/runs/transformer_aug_s42/best.pt`) to `model/best.pt` in the
   Space (i.e. `app/model/best.pt` when running from the project repo).
   Alternatively set the `BSL_CHECKPOINT` environment variable (a Space
   *Variable*) to a checkpoint path.
4. Push; the Space builds and launches automatically.

To run locally from the project repository root:

```bash
python app/app.py
```

with a checkpoint at `app/model/best.pt`, or `BSL_CHECKPOINT` set, or at
least one completed training run under `results/runs/`.

## Data and ethics

Training clips were sourced from publicly available BSL dictionary websites
for academic research; raw videos are not redistributed. The model outputs
only word-level labels from its 50-word vocabulary and stores no user video
beyond the current request.
