# Source Code - Architecture Validation

The code currently in this directory (`data_generation.py`, `model.py`, `train.py`) represents the initial **proof-of-concept phase**. 

As outlined in the Stage 2 Progress Report, this codebase was developed specifically for the **validation of the neural network architecture** and to confirm the mathematical methodology. It runs a localized model over a small subset of samples to demonstrate active gradient reduction.

## Current State
- The training pipeline is functional but discards model weights upon task end.
- Dataset generation is limited for quick validation.
- End-to-end integration with image-based tracking is still pending.

Please refer to `exec_plan.md` in the root directory for the upcoming steps on how this validation code will be scaled up to mass datasets, persistence, and computer vision integration.
