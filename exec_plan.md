# Finalize Four-Bar Linkage ML Synthesis Pipeline

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

## Purpose / Big Picture

While the multi-layer perceptron (MLP) successfully optimizes the inverse mapping of Fourier Descriptors to linkage dimensions, it lacks persistence, visual testing, and image-based input parsing. After completing this plan, a user will be able to:
1. Train the model on mass data and save the resulting checkpoint.
2. Mathematically evaluate the model against untouched testing data and visually verify the predicted mechanism's trajectory. 
3. Draw a continuous trajectory on a canvas (or pass an image), instantly convert it to Fourier Descriptors, and receive the predicted four-bar linkage mechanics that closely mimic that curve.

## Progress

- [x] (Completed) Persist validation model weights and training metadata via `torch.save()` in `src/generate_and_save_validation.py`. Outputs saved to `models/validation_model.pth` and `models/validation_training_meta.pt`.
- [x] (Completed) Generate and store a validation dataset (500 samples) including Fourier descriptors AND raw (x, y) trajectories in `data/validation_dataset.pt`.
- [x] (Completed) Create `scripts/view_data.py` to inspect stored data, plot coupler trajectories, and visualize training loss curves.
- [ ] (Pending) Scale up dataset generation loop to utilize larger `num_samples` securely without memory leaks.
- [ ] (Pending) Create `src/evaluate.py` to parse an unseen validation dataset, run forward kinematics on the ML predictions, and plot the predicted vs. actual Trajectory paths side-by-side. 
- [x] (Completed) Create `src/image_parser.py` (via OpenCV/skimage) to take a monochrome image, extract the continuous coordinate trace boundary, and process it through `compute_fourier_descriptors()`.
- [ ] (Pending) Provide an end-to-end `main.py` script bridging the drawn input to linkage dimensions.

## Surprises & Discoveries

- Observation: None yet recorded.
  Evidence: N/A

## Decision Log

- Decision: Model architecture will remain fixed as a 3-layer ReLU MLP (128 units/layer).
  Rationale: Initial MSRE convergence proves the MLP structure works; tuning should be done on dataset scaling rather than expanding layout needlessly.
  Date/Author: Architecture finalized in `src/model.py` prior to ExecPlan creation.

- Decision: Validation model and data are stored as `.pth` / `.pt` files (PyTorch serialization).
  Rationale: Standard PyTorch convention. `.pth` for model state dicts, `.pt` for general tensor/dict data.
  Date/Author: Established during validation pipeline creation.

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
│   └── generate_and_save_validation.py  # Validation pipeline script
├── data/                   # Stored datasets
│   └── validation_dataset.pt
├── models/                 # Saved model checkpoints
│   ├── validation_model.pth
│   └── validation_training_meta.pt
├── scripts/                # Utility & visualization scripts
│   ├── view_data.py        # Inspect data & plot trajectories/loss
│   ├── generate_plot.py    # Generate training curve plots
│   └── read_docx.py        # Utility for reading docx files
├── tests/                  # Test suite
│   └── test_image_parser.py
├── assets/                 # Generated plots and images
├── docs/                   # Reports, proposals, bibliography
├── README.md
├── exec_plan.md
└── requirements.txt
```

The model pipeline resides in `src/`. `data_generation.py` handles the physical mathematics (forward kinematics) of the four-bar link system. `model.py` hosts the PyTorch network. `train.py` brings them together, optimizing against MSE loss. `generate_and_save_validation.py` runs the full validation pipeline and persists the resulting data and model to disk.

## Plan of Work

1. **Model Persistence**: ~~Modify `src/train.py` to output an artifact.~~ (Completed via `src/generate_and_save_validation.py` → `models/validation_model.pth`)
2. **Evaluation Framework**: Create `src/evaluate.py`. This script will independently load `models/validation_model.pth`, generate a few dozen verification data points using `data_generation.py`, run inference, and plot the actual vs predicted 2D coupler points using Matplotlib.
3. **Computer Vision Input Layer**: ~~Implement `src/image_parser.py`.~~ (Completed)
4. **Integration**: Combine the `image_parser.py` and saved model inference inside a clean interface `main.py` that takes a path via CLI (e.g., `python main.py my_drawing.png`).

## Concrete Steps

1. ~~In `src/train.py`: Uncomment `torch.save` and scale dataset size.~~ (Completed for validation via `generate_and_save_validation.py`)
2. ~~Run validation training: `python src/generate_and_save_validation.py`.~~ (Completed)
3. Create `src/evaluate.py` incorporating Matplotlib subplots for visualization.
4. Run evaluation test: `python src/evaluate.py` (ensure MSE prints correctly and graphs populate).
5. ~~Install `opencv-python` to the `.venv`.~~ (Completed)
6. ~~Draft `src/image_parser.py` with contour-finding logic.~~ (Completed)

## Validation and Acceptance

- **Evaluation Test**: Run `python src/evaluate.py`. You should observe a printed test-loss and a UI window displaying original trajectories drawn directly over ML-predicted trajectories, visually confirming shape conformance.
- **Image Pipeline**: Execute `python main.py path.png`. The console must output valid linkage parameter ratios (a, b, c, d, p_dist, length offset) without crashing. 

## Idempotence and Recovery

Training can be run safely and idempotently as it replaces the previous `.pth` weights. Visual plots do not alter the filesystem. 

## Interfaces and Dependencies

`src/image_parser.py` will require installing `opencv-python` and exporting:
```python
def extract_path_from_image(image_path: str) -> torch.Tensor:
    # Extracts ordered contour pixels and computes FDs.
```

`src/evaluate.py` will leverage existing matplotlib dependencies and export:
```python
def evaluate_model(model_path: str, dataset_size: int = 100):
    # Generates visually rendered plots.
```
