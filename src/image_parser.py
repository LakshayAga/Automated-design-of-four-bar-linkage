import cv2
import numpy as np
import torch
import os

try:
    from data_generation import compute_fourier_descriptors
except ImportError:
    from src.data_generation import compute_fourier_descriptors

def extract_path_from_image(image_path: str) -> torch.Tensor:
    """
    Reads a PNG image with a hand-drawn tracing, extracts its largest contour,
    and computes the normalized Fourier descriptors for the model.
    Assumes standard black drawing on white background.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Load in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}. Check file format.")

    # Apply threshold to invert: we want drawing to be white (255) on black (0) background.
    # A standard black marker on white paper means we invert binary thresholding.
    # So dark pixels (the drawing) become white, light pixels become black.
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours - CHAIN_APPROX_NONE gets all points along the boundary
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        raise ValueError(f"No continuous paths found in {image_path}. Make sure the image contains a clear drawing.")

    # Find the largest contour by length
    largest_contour = max(contours, key=len)

    if len(largest_contour) < 10:
        raise ValueError("The drawn path is too small or noisy to reliably parse.")

    # Extract X and Y coordinates 
    # Contour points have shape (N, 1, 2)
    largest_contour = largest_contour.squeeze(1) # shape (N, 2)
    Px = largest_contour[:, 0].astype(float)
    Py = largest_contour[:, 1].astype(float)

    # Note: Images have the Y axis pointing down. We invert it for Cartesian math.
    max_y = img.shape[0]
    Py = max_y - Py
    
    # Compute the Fourier descriptors (will return 30-element numpy array)
    fd_features = compute_fourier_descriptors(Px, Py, num_descriptors=15)
    
    # Convert numerical array to PyTorch Tensor of float32
    tensor_input = torch.tensor(fd_features, dtype=torch.float32)
    return tensor_input

