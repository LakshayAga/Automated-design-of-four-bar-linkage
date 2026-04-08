# Source Code

The core ML pipeline code for the four-bar linkage synthesis project.

## Modules

| File | Purpose |
|------|---------|
| `data_generation.py` | Forward kinematics, Grashof filtering, Fourier descriptor extraction, and synthetic dataset generation. |
| `model.py` | `LinkagePredictorModel` — 3-layer ReLU MLP (30 → 128 → 128 → 128 → 6). |
| `train.py` | Generic training loop with MSE loss and Adam optimizer. |
| `image_parser.py` | OpenCV-based contour extraction from hand-drawn PNG images → Fourier descriptors. |
| `generate_and_save_validation.py` | Generates validation dataset (500 samples with raw trajectories + FDs), trains the validation model, and persists both to `data/` and `models/`. |

## Stored Artifacts

- **`data/validation_dataset.pt`** — 500 samples: Fourier descriptors `[500, 30]`, normalized linkage params `[500, 6]`, raw coupler trajectories `[500, 128]` (X and Y).
- **`models/validation_model.pth`** — Trained validation model state dict.
- **`models/validation_training_meta.pt`** — Loss history, hyperparameters, architecture metadata.

## Current State
- The validation pipeline is functional and persists model weights and training data to disk.
- Dataset generation captures both Fourier descriptors and raw (x, y) coupler-point coordinates.
- End-to-end integration with image-based tracking is still pending.

Please refer to `exec_plan.md` in the root directory for the upcoming steps.
