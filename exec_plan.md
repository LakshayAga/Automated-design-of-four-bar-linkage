# Finalize Four-Bar Linkage ML Synthesis Pipeline

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

## Purpose / Big Picture

While the multi-layer perceptron (MLP) successfully optimizes the inverse mapping of Fourier Descriptors to linkage dimensions, it lacks persistence, visual testing, and image-based input parsing. After completing this plan, a user will be able to:
1. Train the model on mass data and save the resulting checkpoint.
2. Mathematically evaluate the model against untouched testing data and visually verify the predicted mechanism's trajectory. 
3. Draw a continuous trajectory on a canvas (or pass an image), instantly convert it to Fourier Descriptors, and receive the predicted four-bar linkage mechanics that closely mimic that curve.

## Progress

- [ ] (Pending) Update `train.py` to persist weights via `torch.save()`.
- [ ] (Pending) Scale up dataset generation loop to utilize larger `num_samples` securely without memory leaks.
- [ ] (Pending) Create `evaluate.py` to parse an unseen validation dataset, run forward kinematics on the ML predictions, and plot the predicted vs. actual Trajectory paths side-by-side. 
- [x] (Completed) Create `image_parser.py` (via OpenCV/skimage) to take a monochrome image, extract the continuous coordinate trace boundary, and process it through `compute_fourier_descriptors()`.
- [ ] (Pending) Provide an end-to-end `main.py` script bridging the drawn input to linkage dimensions.

## Surprises & Discoveries

- Observation: None yet recorded.
  Evidence: N/A

## Decision Log

- Decision: Model architecture will remain fixed as a 3-layer ReLU MLP (128 units/layer).
  Rationale: Initial MSRE convergence proves the MLP structure works; tuning should be done on dataset scaling rather than expanding layout needlessly.
  Date/Author: Architecture finalized in `model.py` prior to ExecPlan creation.

## Outcomes & Retrospective

*Pending completion.*

## Context and Orientation

Currently, the model pipeline resides in `src/`. `data_generation.py` handles the physical mathematics (forward kinematics) of the four-bar link system. `model.py` hosts the PyTorch network. `train.py` brings them together, optimizing against MSE loss, but it discards the weights upon task end. The original project scope expects this network to use a strictly image-based, hand-drawn curve input.

## Plan of Work

1. **Model Persistence**: Modify `src/train.py` by uncommenting the `torch.save` payload at the end of the script to output an artifact (e.g. `linkage_model.pth`).
2. **Evaluation Framework**: Create `src/evaluate.py`. This script will independently load `linkage_model.pth`, generate a few dozen verification data points using `data_generation.py`, run inference, and plot the actual vs predicted 2D coupler points using Matplotlib.
3. **Computer Vision Input Layer**: Implement `src/image_parser.py`. Utilize `cv2` to extract edge boundaries of a user-submitted path PNG, order the boundary pixels continuously, and feed them identically to `compute_fourier_descriptors` to yield the required 30-element tensor.
4. **Integration**: Combine the `image_parser.py` and saved model inference inside a clean interface `main.py` that takes a path via CLI (e.g., `python main.py my_drawing.png`).

## Concrete Steps

1. In `src/train.py`:
   - Change `train_model` arguments to handle `dataset_size=50000` to yield physically realistic models.
   - Uncomment line 47: `torch.save(model.state_dict(), "linkage_model.pth")`.
2. Run training: `python src/train.py`.
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
