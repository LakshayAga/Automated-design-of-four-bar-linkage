"""
image_parser.py  --  Fourier descriptor extractor for hand-drawn images
==========================================================================
Extracts the raw pixel contour (original winding order + density from OpenCV)
and computes Fourier Descriptors for use by the ML model.

IMPORTANT: the FD computation must use the RAW pixel coordinates.
Resampling or heavy smoothing before the FFT changes the parameterisation and
pushes the FD values out of the training distribution, degrading model accuracy.
Cosmetic smoothing (for display only) is handled inside inspect_image.py.
"""

import cv2
import numpy as np
import torch
import os

try:
    from data_generation import compute_fourier_descriptors
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from data_generation import compute_fourier_descriptors


def extract_path_from_image(image_path: str) -> torch.Tensor:
    """
    Reads a PNG image with a hand-drawn tracing, extracts its largest contour
    as RAW pixel coordinates, and computes the normalised Fourier descriptors
    for the model.

    Assumes standard black drawing on white background.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}. Check file format.")

    # Invert: dark ink -> white (255), background -> black (0)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError(
            f"No continuous paths found in {image_path}. "
            "Make sure the image contains a clear drawing."
        )

    largest = max(contours, key=len).squeeze(1)   # (N, 2)
    if len(largest) < 10:
        raise ValueError("The drawn path is too small or noisy to reliably parse.")

    Px = largest[:, 0].astype(float)
    Py = img.shape[0] - largest[:, 1].astype(float)   # flip Y to Cartesian

    # Compute FDs from RAW pixel coordinates -- do NOT resample here
    fd_features = compute_fourier_descriptors(Px, Py, num_descriptors=15)

    return torch.tensor(fd_features, dtype=torch.float32)
