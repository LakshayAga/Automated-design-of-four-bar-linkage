# Finalize Four-Bar Linkage ML Synthesis Pipeline

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

## Purpose / Big Picture

While the multi-layer perceptron (MLP) successfully optimizes the inverse mapping of Fourier Descriptors to linkage dimensions, it lacks persistence, visual testing, and image-based input parsing. After completing this plan, a user will be able to:
1. Train the model on mass data and save the resulting checkpoint.
2. Mathematically evaluate the model against untouched testing data and visually verify the predicted mechanism's trajectory. 
3. Draw a continuous trajectory on a canvas (or pass an image), instantly convert it to Fourier Descriptors, and receive the predicted four-bar linkage mechanics that closely mimic that curve.

## Progress

- [x] (Completed) Persist preliminary model weights and training metadata via `torch.save()` in `src/generate_and_save_validation.py`. Outputs saved to `models/preliminary_model.pth` and `models/preliminary_training_meta.pt`.
- [x] (Completed) Generate and store a preliminary dataset (500 samples) including Fourier descriptors AND raw (x, y) trajectories in `data/validation_dataset.pt`.
- [x] (Completed) Create `scripts/view_data.py` to inspect stored data, plot coupler trajectories, and visualize training loss curves.
- [x] (Completed) Replace fake `mse_training_plot.png` with a plot generated from actual stored training metadata.
- [ ] (Pending) Generate full-scale independent datasets: training (10K), validation (2K), and test (500).
- [ ] (Pending) Train production model with validation monitoring and early stopping.
- [ ] (Pending) Create `src/evaluate.py` to run inference on the held-out test set and plot predicted vs. actual trajectory paths side-by-side. 
- [x] (Completed) Create `src/image_parser.py` (via OpenCV/skimage) to take a monochrome image, extract the continuous coordinate trace boundary, and process it through `compute_fourier_descriptors()`.
- [ ] (Pending) Provide an end-to-end `main.py` script bridging the drawn input to linkage dimensions.

## Surprises & Discoveries

- Observation: None yet recorded.
  Evidence: N/A

## Decision Log

- Decision: Model architecture will remain fixed as a 3-layer ReLU MLP (128 units/layer).
  Rationale: Initial MSRE convergence proves the MLP structure works; tuning should be done on dataset scaling rather than expanding layout needlessly.
  Date/Author: Architecture finalized in `src/model.py` prior to ExecPlan creation.

- Decision: Preliminary model and data are stored as `.pth` / `.pt` files (PyTorch serialization).
  Rationale: Standard PyTorch convention. `.pth` for model state dicts, `.pt` for general tensor/dict data.
  Date/Author: Established during preliminary pipeline creation.

- Decision: Rename "validation model" to "preliminary model" to better represent the current state of the model.
  Rationale: The model is trained on a small dataset for pipeline validation purposes only. "Preliminary" more accurately conveys that this is a proof-of-concept, not a validated production model.
  Date/Author: 2026-04-09.

- Decision: No fixed random seed for full-scale dataset generation; instead, log the auto-generated seed in each `.pt` file's metadata.
  Rationale: At scale (10K+ samples), any random draw is statistically representative, so fixing a seed adds no benefit. However, logging the seed that was used allows exact reproduction of any dataset after the fact if needed for debugging or verification. Implementation must call `np.random.get_state()` before generation and store the seed value in the saved `.pt` metadata dict.
  Date/Author: 2026-04-09.

## Outcomes & Retrospective

*Pending completion.*

## Context and Orientation

### Project Structure

```
Project/
├── src/                    # Core source code
│   ├── data_generation.py  # Forward kinematics + dataset generation
│   ├── model.py            # MLP architecture definition
│   ├── train.py            # Training loop (generic)
│   ├── image_parser.py     # OpenCV contour extraction → Fourier descriptors
│   └── generate_and_save_validation.py  # Preliminary pipeline script
├── data/                   # Stored datasets
│   └── validation_dataset.pt
├── models/                 # Saved model checkpoints
│   ├── preliminary_model.pth
│   └── preliminary_training_meta.pt
├── scripts/                # Utility & visualization scripts
│   ├── view_data.py        # Inspect data & plot trajectories/loss
│   ├── generate_plot.py    # Generate training curve from actual stored data
│   └── read_docx.py        # Utility for reading docx files
├── tests/                  # Test suite
│   └── test_image_parser.py
├── assets/                 # Generated plots and images
├── docs/                   # Reports, proposals, bibliography
├── README.md
├── exec_plan.md
└── requirements.txt
```

The model pipeline resides in `src/`. `data_generation.py` handles the physical mathematics (forward kinematics) of the four-bar link system. `model.py` hosts the PyTorch network. `train.py` brings them together, optimizing against MSE loss. `generate_and_save_validation.py` runs the full preliminary pipeline and persists the resulting data and model to disk.

## Plan of Work

1. **Model Persistence**: ~~Modify `src/train.py` to output an artifact.~~ (Completed via `src/generate_and_save_validation.py` → `models/preliminary_model.pth`)
2. **Full Training with Separate Datasets**: Generate independent train/validation/test datasets and train the full model with validation monitoring (see detailed plan below).
3. **Evaluation Framework**: Create `src/evaluate.py`. This script will load the trained model, run inference on the held-out test set, and plot actual vs predicted 2D coupler trajectories.
4. **Computer Vision Input Layer**: ~~Implement `src/image_parser.py`.~~ (Completed)
5. **Integration**: Combine the `image_parser.py` and saved model inference inside a clean interface `main.py` that takes a path via CLI (e.g., `python main.py my_drawing.png`).

---

## Full Training Plan — Separate Datasets

### Motivation

The current preliminary model trains on 100% of its 500 generated samples with no holdout set. This means we have no reliable measure of generalization — the training loss alone cannot confirm whether the model is overfitting. Since we can generate unlimited synthetic data, the cleanest approach is to generate three fully independent datasets.

### Dataset Configuration

| Dataset | Size | Seed | Purpose |
|---------|------|------|---------|
| **Training** | 10,000 | No fixed seed (logged) | Model weight updates. The model learns from this data. |
| **Validation** | 2,000 | No fixed seed (logged) | Monitored after each epoch to detect overfitting. Never used for weight updates. Guides decisions like early stopping. |
| **Test** | 500 | No fixed seed (logged) | Final, one-time evaluation after training is fully complete. Untouched until the very end. Provides the unbiased performance number for reports. |

All three datasets are generated independently by `generate_dataset()` with different random states, ensuring **zero data overlap**. Since no fixed seed is used, each run produces statistically fresh samples. The auto-generated seed is logged in each `.pt` file's metadata so runs remain reproducible after the fact if needed.

### Training Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 100+ | More data warrants longer training; use early stopping to decide when to stop. |
| Batch size | 64 | Larger dataset benefits from larger batches for stable gradient estimates. |
| Learning rate | 1e-3 | Same Adam optimizer as preliminary model. |
| Early stopping patience | 10 | Stop training if validation loss does not improve for 10 consecutive epochs. |

### Implementation Steps

1. **Create `src/generate_full_datasets.py`**:
   - Generate 10,000 training samples → save to `data/train_dataset.pt` (features, targets, trajectories_x, trajectories_y).
   - Generate 2,000 validation samples → save to `data/val_dataset.pt`.
   - Generate 500 test samples → save to `data/test_dataset.pt`.
   - Log the auto-generated random seed in each file’s metadata for reproducibility (see Decision Log).

2. **Create `src/train_full.py`** (or modify `src/train.py`):
   - Load `data/train_dataset.pt` and `data/val_dataset.pt`.
   - Train the model on the training set.
   - After each epoch, compute validation MSE on the validation set (no gradient updates).
   - Record both `train_loss_history` and `val_loss_history`.
   - Implement early stopping: if validation loss has not improved for `patience` epochs, stop training.
   - Save the model weights from the epoch with the best (lowest) validation loss.
   - Output: `models/trained_model.pth` and `models/training_meta.pt` (includes both loss curves, epoch count, hyperparams).

3. **Update `scripts/generate_plot.py`**:
   - Plot training loss AND validation loss on the same axes from `models/training_meta.pt`.

4. **Create `src/evaluate.py`**:
   - Load the trained model and `data/test_dataset.pt`.
   - Run inference on all 500 test samples.
   - Print final test MSE.
   - Plot a selection of predicted vs actual coupler-point trajectories side-by-side.

### Expected Output

```
Generating datasets...
  Training:   10,000 samples -> data/train_dataset.pt
  Validation:  2,000 samples -> data/val_dataset.pt
  Test:          500 samples -> data/test_dataset.pt

Training...
  Epoch   1/200 | Train MSE: 5.8421 | Val MSE: 5.9102
  Epoch  10/200 | Train MSE: 0.4512 | Val MSE: 0.5034
  Epoch  50/200 | Train MSE: 0.1823 | Val MSE: 0.2145
  Epoch  87/200 | Train MSE: 0.0942 | Val MSE: 0.1203
  Early stopping at epoch 97 (no improvement for 10 epochs).
  Best epoch: 87 | Val MSE: 0.1203

Final Test Evaluation:
  Test MSE: 0.1185
```

### Validation Criteria

- The gap between training MSE and validation MSE should remain small throughout training, indicating the model generalizes well.
- Validation loss should decrease consistently before early stopping triggers.
- Final test MSE should be comparable to the best validation MSE, confirming the validation set was a reliable proxy.

---

## Concrete Steps

1. ~~In `src/train.py`: Uncomment `torch.save` and scale dataset size.~~ (Completed for preliminary via `generate_and_save_validation.py`)
2. ~~Run preliminary training: `python src/generate_and_save_validation.py`.~~ (Completed)
3. Generate full datasets: `python src/generate_full_datasets.py`.
4. Train full model with validation monitoring: `python src/train_full.py`.
5. Create `src/evaluate.py` incorporating Matplotlib subplots for visualization.
6. Run evaluation on test set: `python src/evaluate.py` (ensure MSE prints correctly and trajectory plots populate).
7. ~~Install `opencv-python` to the `.venv`.~~ (Completed)
8. ~~Draft `src/image_parser.py` with contour-finding logic.~~ (Completed)

## Validation and Acceptance

- **Training Test**: Run the full training script. Verify that the generated plot shows both training and validation loss curves, and that early stopping triggers at a reasonable epoch.
- **Evaluation Test**: Run `python src/evaluate.py`. You should observe a printed test MSE and a window displaying original trajectories drawn directly over ML-predicted trajectories, visually confirming shape conformance.
- **Image Pipeline**: Execute `python main.py path.png`. The console must output valid linkage parameter ratios (a, b, c, d, p_dist, length offset) without crashing. 

## Idempotence and Recovery

Training can be run safely and idempotently as it replaces the previous `.pth` weights. Visual plots do not alter the filesystem. Dataset generation is also idempotent — rerunning produces fresh data (old files are overwritten).

## Interfaces and Dependencies

`src/image_parser.py` will require installing `opencv-python` and exporting:
```python
def extract_path_from_image(image_path: str) -> torch.Tensor:
    # Extracts ordered contour pixels and computes FDs.
```

`src/evaluate.py` will leverage existing matplotlib dependencies and export:
```python
def evaluate_model(model_path: str, test_data_path: str):
    # Loads test set, runs inference, plots trajectories.
```

