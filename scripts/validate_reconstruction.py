"""
validate_reconstruction.py -- Batch reconstruction error evaluation over the test set.

For each test sample: runs the model on the stored Fourier descriptors, reconstructs
the coupler path via forward kinematics, and computes the symmetric Hausdorff distance
between the (normalised) ground-truth trajectory and the reconstructed one.

Outputs a histogram saved to assets/reconstruction_error_hist.png.

Usage (from project root):
    python scripts/validate_reconstruction.py
    python scripts/validate_reconstruction.py --dataset data/test_dataset.pt
    python scripts/validate_reconstruction.py --model   models/best_model.pth

NOTE: For single-image inference use  scripts/run_inference.py  instead.
"""

import os
import sys
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from data_generation import forward_kinematics_trajectory
from predict import DEFAULT_MODEL, load_model

BG     = "#f8fafc"
DARK   = "#1e293b"
ACCENT = "#2563eb"
GREEN  = "#16a34a"
RED    = "#dc2626"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": "#cbd5e1", "axes.labelcolor": DARK,
    "xtick.color": DARK, "ytick.color": DARK,
    "text.color": DARK, "grid.color": "#e2e8f0",
    "font.family": "sans-serif",
})


def hausdorff_sym(A: np.ndarray, B: np.ndarray) -> float:
    """Symmetric Hausdorff distance between two (N,2) point sets."""
    return max(directed_hausdorff(A, B)[0], directed_hausdorff(B, A)[0])


def center_scale(Px, Py):
    Px = Px - Px.mean(); Py = Py - Py.mean()
    scale = max((Px.max() - Px.min()), (Py.max() - Py.min()))
    if scale > 1e-9:
        Px /= scale; Py /= scale
    return Px, Py


def reconstruct_from_pred(pred_tensor: torch.Tensor, num_points: int = 256):
    """pred_tensor: [6] float.  Returns (Px, Py) or None if degenerate."""
    p  = pred_tensor.numpy()
    nd = p[3]
    if nd < 1e-9:
        return None
    a, b, c, d = p[0]/nd, p[1]/nd, p[2]/nd, 1.0
    pp, phi    = p[4]/nd, p[5]
    return forward_kinematics_trajectory(a, b, c, d, pp, phi, num_points=num_points)


def main():
    parser = argparse.ArgumentParser(
        description="Batch reconstruction-error evaluation on the test set."
    )
    parser.add_argument(
        "--dataset", default="data/test_dataset.pt",
        help="Path to .pt dataset (default: data/test_dataset.pt)."
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Path to model checkpoint (default: models/best_model.pth)."
    )
    args = parser.parse_args()

    dataset_path = (
        args.dataset if os.path.isabs(args.dataset)
        else os.path.join(PROJECT_ROOT, args.dataset)
    )
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found: {dataset_path}")
        sys.exit(1)

    print(f"\nLoading dataset  : {dataset_path}")
    data  = torch.load(dataset_path, weights_only=True)
    feats = data["features"]          # [N, 30]
    gt    = data["targets"]           # [N, 6]
    tx    = data["trajectories_x"]    # [N, 128]
    ty    = data["trajectories_y"]    # [N, 128]
    N     = feats.shape[0]

    print(f"Loading model    : {args.model}")
    model = load_model(args.model)

    print(f"Running inference on {N} samples ...\n")
    with torch.no_grad():
        preds = model(feats)          # [N, 6]

    errors = []
    failed = 0

    for i in range(N):
        # Ground truth trajectory (from simulator)
        inp_x = tx[i].numpy()
        inp_y = ty[i].numpy()
        cx, cy = center_scale(inp_x.copy(), inp_y.copy())
        inp_path = np.column_stack([cx, cy])

        # Reconstructed trajectory from model prediction
        res = reconstruct_from_pred(preds[i])
        if res is None:
            failed += 1
            continue
        rPx, rPy = res
        if rPx is None or np.any(np.isnan(rPx)):
            failed += 1
            continue

        rx, ry = center_scale(rPx.copy(), rPy.copy())
        errors.append(hausdorff_sym(inp_path, np.column_stack([rx, ry])))

    errors = np.array(errors)

    sep = "=" * 60
    print(sep)
    print(f"  Batch Reconstruction Validation")
    print(sep)
    print(f"  Dataset          : {dataset_path}")
    print(f"  Model            : {args.model}")
    print(f"  Total samples    : {N}")
    print(f"  FK succeeded     : {N - failed}   ({failed} degenerate / failed)")
    if len(errors) > 0:
        print(f"  Hausdorff mean   : {errors.mean():.4f}")
        print(f"  Hausdorff median : {np.median(errors):.4f}")
        print(f"  Hausdorff min    : {errors.min():.4f}")
        print(f"  Hausdorff max    : {errors.max():.4f}")
    print(sep)

    # Histogram
    if len(errors) > 0:
        fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG)
        ax.set_facecolor(BG)
        ax.hist(errors, bins=min(20, len(errors)), color=ACCENT, edgecolor="white", alpha=0.9)
        ax.axvline(errors.mean(),       color=RED,   linestyle="--", linewidth=1.8,
                   label=f"Mean   = {errors.mean():.3f}")
        ax.axvline(np.median(errors),   color=GREEN, linestyle="--", linewidth=1.8,
                   label=f"Median = {np.median(errors):.3f}")
        ax.set_xlabel("Hausdorff Distance (normalised units)")
        ax.set_ylabel("Count")
        ax.set_title("Reconstruction Error Distribution", fontweight="bold", color=DARK)
        ax.legend(fontsize=10)

        assets_dir = os.path.join(PROJECT_ROOT, "assets")
        os.makedirs(assets_dir, exist_ok=True)
        out = os.path.join(assets_dir, "reconstruction_error_hist.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Histogram saved : {out}")


if __name__ == "__main__":
    main()
