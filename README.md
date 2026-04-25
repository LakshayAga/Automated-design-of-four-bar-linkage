# Automated Design of Four-Bar Linkage from Hand-Drawn Coupler Trajectories

A deep learning system that automatically predicts the physical dimensions of a planar **crank-rocker four-bar linkage** from a hand-drawn coupler path. The user sketches a desired trajectory; the system outputs the six link parameters needed to build a mechanism that traces it.

---

## Table of Contents

1. [Background](#1-background)
2. [How It Works](#2-how-it-works)
3. [Project Structure](#3-project-structure)
4. [Installation](#4-installation)
5. [Usage](#5-usage)
6. [Model Architecture](#6-model-architecture)
7. [Training](#7-training)
8. [Data Generation](#8-data-generation)
9. [Inference Pipeline](#9-inference-pipeline)
10. [Output Parameters](#10-output-parameters)
11. [References](#11-references)

---

## 1. Background

Designing a planar four-bar linkage to follow a specific path is a classical challenge in mechanical engineering. Traditional graphical or analytical methods are tedious and limited in accuracy. This project replaces them with a supervised neural network trained on a large synthetic dataset of crank-rocker mechanisms and their coupler point paths.

A **crank-rocker** is a specific class of four-bar linkage in which:
- The **crank** (`a`) rotates continuously (shortest link, adjacent to ground)
- The **rocker** (`c`) oscillates back and forth
- The **coupler** (`b`) connects them, carrying a point (`p`) that traces the desired path
- The **ground** (`d`) is the fixed frame link

The model is restricted to crank-rocker mechanisms only (a stricter subset of Grashof linkages), which narrows the solution space and improves prediction accuracy.

---

## 2. How It Works

The end-to-end pipeline has three stages:

```
Hand-drawn image
      │
      ▼
┌─────────────────────┐
│   Image Parser      │  OpenCV contour extraction → raw pixel (Px, Py)
│   (image_parser.py) │  Fourier Descriptors → 30-element feature vector
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│   Neural Network    │  Z-score normalise FD → MLP → 6 predicted parameters
│   (model.py)        │  Runs in milliseconds
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│   Local Optimiser   │  Nelder-Mead refines prediction by minimising
│   (refine.py)       │  FD-magnitude + aspect-ratio distance
└─────────────────────┘
      │
      ▼
   norm_a, norm_b, norm_c, norm_d, norm_p, p_angle
   (six parameters defining the crank-rocker linkage)
```

### Why Fourier Descriptors?

Raw pixel coordinates are variable-length and sensitive to where the drawing starts. Fourier Descriptors (FDs) convert the path into a fixed-size frequency-domain representation that is:
- **Translation-invariant** (DC component zeroed)
- **Scale-invariant** (normalised by |F₁|)
- **Fixed-length** (15 complex descriptors → 30 real values)

This means the model learns pure shape, independent of where or how large the drawing is.

---

## 3. Project Structure

```
project/
├── src/
│   ├── data_generation.py          # Forward kinematics, Grashof/crank-rocker checks,
│   │                               # Fourier descriptor computation, dataset generation
│   ├── model.py                    # MLP architecture (LinkagePredictorModel)
│   ├── train.py                    # Lightweight prototype training script
│   ├── train_full.py               # Full training: validation, early stopping,
│   │                               # wrapped angle loss, LR scheduling
│   ├── generate_full_dataset.py    # Generates train/val/test splits (75k/15k/3k)
│   │                               # with z-score normalisation
│   ├── image_parser.py             # OpenCV contour → Fourier descriptors
│   ├── predict.py                  # Inference module: image → predicted parameters
│   ├── reconstruct.py              # Forward kinematics reconstruction + validation
│   └── refine.py                   # Nelder-Mead post-prediction optimisation
├── scripts/
│   └── run_inference.py            # End-to-end inference script (main entry point)
├── data/
│   ├── train_dataset.pt            # 75,000 crank-rocker samples
│   ├── val_dataset.pt              # 15,000 samples
│   ├── test_dataset.pt             # 3,000 samples
│   └── feature_norm.pt             # Training-set FD mean/std (used at inference)
├── models/
│   ├── best_model.pth              # Best model checkpoint (lowest val loss)
│   └── best_model_meta.pt          # Training history and hyperparameters
├── inputs/                         # Drop hand-drawn images here
├── runs/                           # Inference outputs (one folder per image)
├── requirements.txt
└── README.md
```

---

## 4. Installation

**Requirements:** Python 3.9+

```bash
git clone https://github.com/LakshayAga/Automated-design-of-four-bar-linkage.git
cd Automated-design-of-four-bar-linkage
pip install -r requirements.txt
```

Key dependencies: `torch`, `numpy`, `scipy`, `opencv-python`, `matplotlib`, `tqdm`

---

## 5. Usage

### Quick Inference (recommended)

1. Drop your hand-drawn path image (PNG or JPG, black drawing on white background) into the `inputs/` folder.
2. Run from the project root:

```bash
# Auto-detects the image in inputs/
python scripts/run_inference.py

# Specify a filename
python scripts/run_inference.py my_drawing.png

# Skip local optimisation (faster, model prediction only)
python scripts/run_inference.py my_drawing.png --no-refine

# Use a different model checkpoint
python scripts/run_inference.py my_drawing.png --model models/my_checkpoint.pth

# Control optimisation effort
python scripts/run_inference.py my_drawing.png --refine-steps 3000 --restarts 8
```

3. Results are saved to `runs/<image_name>/`:

| File | Contents |
|---|---|
| `params.json` | Raw model-predicted linkage parameters |
| `params_refined.json` | Parameters after Nelder-Mead optimisation |
| `reconstruction.png` | Side-by-side comparison: input path vs model vs refined |
| `fd_diagnostic.png` | Image-parser diagnostic (contour extraction quality check) |
| `summary.txt` | Human-readable report with all parameters and improvement % |

### Training from Scratch

**Step 1 — Generate the dataset:**
```bash
python src/generate_full_dataset.py
# Optional: customise sizes
python src/generate_full_dataset.py --train 75000 --val 15000 --test 3000
```

**Step 2 — Train the model:**
```bash
python src/train_full.py
# Optional: customise hyperparameters
python src/train_full.py --epochs 1000 --lr 3e-4 --batch 128 --patience 40
```

The best model checkpoint is automatically saved to `models/best_model.pth`.

---

## 6. Model Architecture

The neural network is a **Multi-Layer Perceptron (MLP)** implemented in PyTorch.

```
Input [30]  ← z-score normalised Fourier Descriptors, clipped to [-5, 5]
   │
   ├─ Linear(30 → 256) → Tanh
   ├─ Linear(256 → 256) → Tanh
   └─ Linear(256 → 256) → Tanh
                │
        ┌───────┴────────┐
        │                │
   ratio_head        angle_head
   Linear(256→5)     Linear(256→1)
   0.05+sigmoid×0.95  tanh(x)×π
   → (0.05, 1.0)      → (-π, π)
        │                │
        └───────┬────────┘
             concat
          Output [6]
```

**Total parameters: 141,062**

| Design Choice | Reason |
|---|---|
| Tanh backbone (not ReLU) | Matches normalised [-1,1] input scale; keeps activations bounded going into the output head |
| Hidden dim 256 | Sufficient capacity for the crank-rocker-only problem space |
| Decoupled output head | Ratios and angles have fundamentally different ranges and need different activations |
| `0.05 + sigmoid(x) × 0.95` | Hard-clamps ratios to (0.05, 1.0), matching training data bounds (min link 0.5, max 10.0 → min ratio 0.05) |
| `tanh(x) × π` for angle | Naturally bounds the output to (-π, π) |

---

## 7. Training

### Loss Function

Plain MSE is not used. The loss is split into two physically meaningful terms:

```
Total Loss = MSE(predicted ratios, true ratios)
           + 0.5 × WrappedAngleMSE(predicted angle, true angle)
```

The **wrapped angle MSE** uses:
```python
diff = atan2(sin(pred - true), cos(pred - true))
loss = mean(diff²)
```

This correctly handles the (-π, π) boundary — the angular distance between 179° and -179° is 2°, not 358°. Without wrapping, the model would be penalised heavily for predictions near the boundary even when they are geometrically correct.

### Training Configuration (defaults)

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 3e-4 |
| Batch size | 128 |
| Max epochs | 1000 |
| Early stopping patience | 40 epochs |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Angle loss weight | 0.5 |
| Random seed | 42 |

Early stopping monitors validation loss and halts training if no improvement is seen for 40 consecutive epochs. The scheduler halves the learning rate every 5 epochs without improvement, allowing fine-grained convergence once the model is close to a minimum.

---

## 8. Data Generation

### Mechanism Filter

All samples must satisfy the **crank-rocker condition**:
1. Grashof condition holds: `S + L ≤ P + Q`
2. Link `a` is the shortest link: `a ≤ b`, `a ≤ c`, `a ≤ d`

This guarantees `a` rotates fully (crank) and `c` oscillates (rocker).

### Forward Kinematics

The coupler point path is computed analytically using **Freudenstein's equation**. For each crank angle θ₂ in [0, 2π], the rocker angle θ₄ and coupler angle θ₃ are solved, then the coupler point position is:

```
Px = a·cos(θ₂) + p·cos(θ₃ + φ)
Py = a·sin(θ₂) + p·sin(θ₃ + φ)
```

where `p` is the coupler point offset distance and `φ` is its angle relative to the coupler link.

### Normalisation

Link lengths are sampled uniformly in [0.5, 10.0]. Targets are stored as **ratios relative to the longest link**, making the model scale-invariant.

Fourier Descriptor features are **z-score normalised** using training-set statistics only (no data leakage), then clipped to [-5, 5] to prevent Tanh saturation on outlier inputs. The normalisation statistics are saved to `data/feature_norm.pt` and applied identically at inference time.

### Dataset Sizes

| Split | Samples | Seed |
|---|---|---|
| Train | 75,000 | 42 |
| Val | 15,000 | 43 |
| Test | 3,000 | 44 |

---

## 9. Inference Pipeline

### Image Parsing (`image_parser.py`)

Input images are expected to be black drawings on a white background. Processing:
1. Convert to grayscale and threshold (invert so ink = 255)
2. Extract contours using `cv2.CHAIN_APPROX_NONE` (keeps all contour points)
3. Select the largest contour
4. Flip Y-axis to convert from image coordinates to Cartesian
5. Compute Fourier Descriptors from **raw pixel coordinates** — no resampling or smoothing, as this would alter the FD values and push inputs out of the training distribution

### Post-Prediction Refinement (`refine.py`)

The neural network provides a fast initial estimate. A local optimiser then refines it by directly minimising path similarity. The objective function combines three terms:

**FD magnitude distance** — Compares only the magnitudes |F_k| of the Fourier spectra, not the raw complex values. This is phase-invariant (the traversal starting point of the curve does not affect the result).

**Aspect-ratio penalty** — Compares the height/width ratio of the two paths using a PCA-aligned bounding box (rotation-invariant). Prevents degenerate solutions that have low FD distance but wrong overall shape.

**Link-ratio penalty** — Penalises extreme link length ratios (>3×), which tend to produce near-open-arc degenerate paths.

The optimiser is **Nelder-Mead** (derivative-free, suitable for low-dimensional problems with non-smooth objectives). Parameters are encoded via logit/sigmoid to allow unconstrained optimisation while keeping outputs in valid bounded ranges. Multiple random restarts are used to escape local minima.

### Validity Checking (`reconstruct.py`)

Before running forward kinematics on any predicted parameter set, validity is checked:
- All link ratios ≥ 0.05
- Grashof condition satisfied
- Reconstructed path has no NaN values
- Path spans more than a degenerate point

---

## 10. Output Parameters

All outputs are normalised relative to the longest link in the assembly.

| Parameter | Symbol | Range | Meaning |
|---|---|---|---|
| `norm_a` | a / max\_link | (0.05, 1.0) | Crank length ratio (always smallest) |
| `norm_b` | b / max\_link | (0.05, 1.0) | Coupler length ratio |
| `norm_c` | c / max\_link | (0.05, 1.0) | Rocker length ratio |
| `norm_d` | d / max\_link | (0.05, 1.0) | Ground link length ratio |
| `norm_p` | p / max\_link | (0.05, 1.0) | Coupler point offset ratio |
| `p_angle` | φ | (-π, π) | Coupler point angle (radians) |

To recover absolute dimensions, choose any desired scale for the longest link and multiply all ratios by it. The ground link `d` is used as the internal reference during reconstruction (`d = 1.0`).

---

## 11. References

1. B. Röder, S. Hajipour, H. Ebel, P. Eberhard, and D. Bestle, "Automated design of a four-bar mechanism starting from hand drawings of desired coupler trajectories and velocity profiles," *Mechanics Based Design of Structures and Machines*, 2025.
2. C. M. Bishop, *Pattern Recognition and Machine Learning*. Springer-Verlag, 2006.
3. L. Herrmann, M. Jokeit, O. Weeger, and S. Kollmannsberger, *Deep Learning in Computational Mechanics: An Introductory Course*. Springer Nature, 2025.
