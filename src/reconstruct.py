"""
reconstruct.py -- Shared reconstruction utilities used by run_inference and validate_reconstruction.

Provides:
    reconstruct_fk(params_dict, num_points)  -> (Px, Py) or None
    center_scale(Px, Py)                     -> (Px, Py)
    check_params_validity(params_dict)       -> (ok: bool, reason: str)
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_generation import forward_kinematics_trajectory, is_grashof

# Minimum valid ratio — matches training data: min_link=0.5, max_link=10.0 => 0.05
MIN_RATIO = 0.05


def check_params_validity(p: dict) -> tuple[bool, str]:
    """
    Check whether predicted parameters will produce a usable FK result.
    Returns (True, "OK") or (False, reason_string).
    """
    for key in ["norm_a", "norm_b", "norm_c", "norm_d", "norm_p"]:
        if p[key] < MIN_RATIO:
            return False, f"{key}={p[key]:.4f} is below minimum ({MIN_RATIO}) — near-zero link length"

    nd = p["norm_d"]
    if nd < MIN_RATIO:
        return False, f"norm_d={nd:.4f} too small — cannot normalise to reference frame"

    a = p["norm_a"] / nd
    b = p["norm_b"] / nd
    c = p["norm_c"] / nd
    d = 1.0
    if not is_grashof(a, b, c, d):
        return False, (
            f"Grashof condition violated: S+L > P+Q "
            f"(a={a:.3f}, b={b:.3f}, c={c:.3f}, d={d:.3f})"
        )

    return True, "OK"


def reconstruct_fk(p: dict, num_points: int = 256):
    """
    Run forward kinematics from a params dict (output of predict_linkage).
    Uses norm_d as reference (d_abs = 1.0).
    Returns (Px, Py) arrays, or None on failure.
    """
    ok, reason = check_params_validity(p)
    if not ok:
        return None, reason

    nd = p["norm_d"]
    a, b, c = p["norm_a"]/nd, p["norm_b"]/nd, p["norm_c"]/nd
    d = 1.0
    pp, phi = p["norm_p"]/nd, p["p_angle"]

    result = forward_kinematics_trajectory(a, b, c, d, pp, phi, num_points=num_points)
    if result is None:
        return None, "Forward kinematics returned None (singular configuration)"

    Px, Py = result
    if np.any(np.isnan(Px)) or np.any(np.isnan(Py)):
        return None, "Forward kinematics produced NaN values"

    # Sanity check: path must span more than a point
    span = max(Px.max() - Px.min(), Py.max() - Py.min())
    if span < 1e-4:
        return None, f"Reconstructed path is near-degenerate (span={span:.2e}) — crank too small"

    return (Px, Py), "OK"


def center_scale(Px: np.ndarray, Py: np.ndarray):
    """Centre and normalise a path to a unit bounding box."""
    Px = Px - Px.mean()
    Py = Py - Py.mean()
    scale = max(Px.max() - Px.min(), Py.max() - Py.min())
    if scale > 1e-9:
        Px /= scale
        Py /= scale
    return Px, Py
