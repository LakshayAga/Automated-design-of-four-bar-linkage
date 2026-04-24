"""
inspect_image.py - Image Parser Validation Tool
================================================
Loads a hand-drawn linkage trace image, extracts the largest contour,
computes its Fourier descriptors, and produces a 4-panel diagnostic figure:

  Panel 1 – Original grayscale image
  Panel 2 – Binary (thresholded) image used for contour detection
  Panel 3 – Extracted contour path in Cartesian coordinates
  Panel 4 – Fourier descriptor magnitudes (first N coefficients)

Usage (from project root):
    python scripts/inspect_image.py path/to/your_image.png
    python scripts/inspect_image.py path/to/your_image.png --num_descriptors 20
    python scripts/inspect_image.py path/to/your_image.png --save
"""

import sys
import os
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter1d

# ── Path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from data_generation import compute_fourier_descriptors

# ── Styling ─────────────────────────────────────────────────────────────────
ACCENT  = "#2563eb"   # blue
GREEN   = "#16a34a"
RED     = "#dc2626"
ORANGE  = "#ea580c"
BG      = "#f8fafc"
DARK    = "#1e293b"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "axes.edgecolor":   "#cbd5e1",
    "axes.labelcolor":  DARK,
    "xtick.color":      DARK,
    "ytick.color":      DARK,
    "text.color":       DARK,
    "grid.color":       "#e2e8f0",
    "font.family":      "sans-serif",
})


# ── Smoothing helpers ────────────────────────────────────────────────────────
_DISPLAY_RESAMPLE_N = 512         # arc-length resample target for DISPLAY only
_DISPLAY_WINDOW     = 15          # light circular moving-average for display


def _resample_arc(Px, Py, N=_DISPLAY_RESAMPLE_N):
    """Resample a closed curve to N equally-spaced points along its arc length."""
    cx = np.append(Px, Px[0])
    cy = np.append(Py, Py[0])
    ds  = np.sqrt(np.diff(cx)**2 + np.diff(cy)**2)
    arc = np.concatenate([[0], np.cumsum(ds)])
    t   = np.linspace(0, arc[-1], N, endpoint=False)
    return np.interp(t, arc, cx), np.interp(t, arc, cy)


def smooth_contour(Px, Py, window=_DISPLAY_WINDOW, resample_n=_DISPLAY_RESAMPLE_N):
    """
    Produce a smoothed version of the contour for DISPLAY purposes only.
    NOT used for Fourier descriptor computation -- raw pixel coordinates are
    used for that to preserve the FD parameterisation the model was trained on.

    Steps:
      1. Resample to `resample_n` arc-length-equidistant points
         (removes pixel-density clustering near curves).
      2. Apply a light circular moving-average to eliminate pen-overlap kinks
         at the loop closure without distorting overall shape.
    """
    Px_rs, Py_rs = _resample_arc(Px, Py, resample_n)
    if window > 1:
        w = window
        def cma(a):
            ext = np.concatenate([a[-w:], a, a[:w]])
            return uniform_filter1d(ext, size=w)[w:-w]
        Px_rs, Py_rs = cma(Px_rs), cma(Py_rs)
    return Px_rs, Py_rs


# ── Core extraction ──────────────────────────────────────────────────────────
def extract_contour(image_path: str):
    """
    Returns (img_gray, thresh, Px, Py, contours) where:
      img_gray  -- original grayscale image (H x W uint8)
      thresh    -- binary thresholded image (H x W uint8, drawing = 255)
      Px, Py    -- SMOOTHED contour for DISPLAY (arc-resampled 512 pts, w=15 MA)
                   used in panel 3 and the reconstruction overlay plot
      contours  -- raw OpenCV contours list

    Px_raw / Py_raw (raw pixel contour) are what should be fed to
    compute_fourier_descriptors() and predict_linkage() so the FD
    parameterisation matches the training data exactly.
    Call extract_contour_raw() if you need the raw arrays.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Invert: drawing (dark ink) -> white (255), background -> black (0)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contours found. Make sure the image has a clear dark line on a white background.")

    largest = max(contours, key=len).squeeze(1)   # (N, 2)
    if len(largest) < 10:
        raise ValueError("Contour too small -- check image quality.")

    Px_raw = largest[:, 0].astype(float)
    Py_raw = largest[:, 1].astype(float)

    # Flip Y so +Y points up (Cartesian convention)
    Py_raw = img.shape[0] - Py_raw

    # Smooth ONLY for display -- raw values are used for FD/model
    Px_disp, Py_disp = smooth_contour(Px_raw, Py_raw)

    return img, thresh, Px_disp, Py_disp, contours


def extract_contour_raw(image_path: str):
    """
    Like extract_contour() but returns the RAW (un-smoothed) pixel coordinates.
    These must be used when computing Fourier descriptors for the model,
    since the model was trained on FDs from raw FK trajectories whose
    parameterisation matches raw pixel traversal better than arc-length resampling.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contours found.")

    largest = max(contours, key=len).squeeze(1)
    Px = largest[:, 0].astype(float)
    Py = img.shape[0] - largest[:, 1].astype(float)
    return Px, Py


def build_figure(img, thresh, Px, Py, contours, fd_features, num_descriptors, image_path):
    """Build and return the 4-panel diagnostic matplotlib figure."""

    fig = plt.figure(figsize=(16, 8), facecolor=BG)
    fig.suptitle(
        f"Image Parser Diagnostic  ·  {os.path.basename(image_path)}",
        fontsize=14, fontweight="bold", color=DARK, y=0.98
    )

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.42, wspace=0.38)

    # ── Panel 1: Original image ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(img, cmap="gray", aspect="equal")
    ax1.set_title("1. Original Image", fontweight="bold")
    ax1.axis("off")

    # ── Panel 2: Thresholded + contour overlay ───────────────────────────────
    ax2 = fig.add_subplot(gs[:, 1])
    thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    # Draw all contours in red, largest in blue
    cv2.drawContours(thresh_rgb, contours, -1, (220, 38, 38), 2)
    largest_c = max(contours, key=len)
    cv2.drawContours(thresh_rgb, [largest_c], -1, (37, 99, 235), 2)
    ax2.imshow(thresh_rgb, aspect="equal")
    ax2.set_title("2. Binary + Detected Contour\n(blue = largest, selected)", fontweight="bold")
    ax2.axis("off")

    # ── Panel 3: Extracted path in Cartesian space ───────────────────────────
    ax3 = fig.add_subplot(gs[:, 2])
    n_pts = len(Px)
    colors = plt.cm.viridis(np.linspace(0, 1, n_pts))
    ax3.scatter(Px, Py, c=colors, s=4, zorder=3, alpha=0.85)
    ax3.plot(Px, Py, color=ACCENT, linewidth=1.0, alpha=0.5, zorder=2)
    ax3.plot(Px[0], Py[0], "o", color=GREEN, markersize=9, label=f"Start (pt 0)", zorder=5)
    ax3.plot(Px[-1], Py[-1], "s", color=ORANGE, markersize=7, label=f"End (pt {n_pts-1})", zorder=5)
    ax3.set_title(f"3. Extracted Path\n({n_pts} points, Y-axis flipped to Cartesian)", fontweight="bold")
    ax3.set_xlabel("X (px)")
    ax3.set_ylabel("Y (px, flipped)")
    ax3.set_aspect("equal")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(fontsize=8)

    # ── Panel 4: Fourier Descriptor magnitudes ───────────────────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    # fd_features is [re0, im0, re1, im1, ...] length = num_descriptors*2
    reals = fd_features[0::2]
    imags = fd_features[1::2]
    mags  = np.sqrt(reals**2 + imags**2)
    x_idx = np.arange(1, num_descriptors + 1)
    ax4.bar(x_idx, mags, color=ACCENT, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax4.set_title("4a. FD Magnitudes |Fk|", fontweight="bold")
    ax4.set_xlabel("Descriptor index k")
    ax4.set_ylabel("|Fk|")
    ax4.grid(True, axis="y", linestyle="--", alpha=0.5)

    # ── Panel 5: FD real & imaginary components ──────────────────────────────
    ax5 = fig.add_subplot(gs[1, 3])
    width = 0.38
    ax5.bar(x_idx - width/2, reals, width=width, color=GREEN,  alpha=0.85, label="Real",      edgecolor="white", linewidth=0.5)
    ax5.bar(x_idx + width/2, imags, width=width, color=RED,    alpha=0.80, label="Imaginary", edgecolor="white", linewidth=0.5)
    ax5.axhline(0, color=DARK, linewidth=0.8, linestyle="--")
    ax5.set_title("4b. FD Real & Imag Parts", fontweight="bold")
    ax5.set_xlabel("Descriptor index k")
    ax5.set_ylabel("Value")
    ax5.legend(fontsize=8)
    ax5.grid(True, axis="y", linestyle="--", alpha=0.5)

    return fig


def print_fd_table(fd_features, num_descriptors):
    """Pretty-print FD values to the console."""
    reals = fd_features[0::2]
    imags = fd_features[1::2]
    mags  = np.sqrt(reals**2 + imags**2)

    print("\n" + "=" * 62)
    print(f"  Fourier Descriptors  (first {num_descriptors} coefficients)")
    print("=" * 62)
    print(f"  {'k':>4}  {'Real':>12}  {'Imaginary':>12}  {'Magnitude':>12}")
    print("  " + "-" * 56)
    for k in range(num_descriptors):
        print(f"  {k+1:>4}  {reals[k]:>12.4f}  {imags[k]:>12.4f}  {mags[k]:>12.4f}")
    print("=" * 62)
    print(f"\n  Raw feature vector (length {len(fd_features)}):")
    print("  " + np.array2string(fd_features, formatter={"float_kind": lambda x: f"{x:8.4f}"}, separator=", "))
    print()


# ── CLI entry point ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Inspect image parser output: contour + Fourier descriptors."
    )
    parser.add_argument("image_path", help="Path to a PNG/JPG image with a hand-drawn linkage trace.")
    parser.add_argument(
        "--num_descriptors", "-n", type=int, default=15,
        help="Number of Fourier descriptor pairs to compute (default: 15, matching training data)."
    )
    parser.add_argument(
        "--save", "-s", action="store_true",
        help="Save the figure to assets/inspect_<imagename>.png instead of showing it interactively."
    )
    args = parser.parse_args()

    image_path = os.path.abspath(args.image_path)
    if not os.path.isfile(image_path):
        print(f"[ERROR] File not found: {image_path}")
        sys.exit(1)

    print(f"\n[inspect_image] Loading: {image_path}")
    img, thresh, Px, Py, contours = extract_contour(image_path)
    print(f"  -> Contour extracted: {len(Px)} points")

    fd_features = compute_fourier_descriptors(Px, Py, num_descriptors=args.num_descriptors)
    print(f"  -> Fourier descriptors computed: {len(fd_features)} values ({args.num_descriptors} complex pairs)")

    print_fd_table(fd_features, args.num_descriptors)

    fig = build_figure(img, thresh, Px, Py, contours, fd_features, args.num_descriptors, image_path)

    if args.save:
        assets_dir = os.path.join(PROJECT_ROOT, "assets")
        os.makedirs(assets_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(assets_dir, f"inspect_{stem}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[inspect_image] Figure saved to: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
