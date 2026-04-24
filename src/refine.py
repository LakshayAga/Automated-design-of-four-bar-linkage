"""
refine.py -- Post-prediction local optimisation.

After the neural network makes an initial prediction, this module refines it
using Nelder-Mead (derivative-free) to minimise the Fourier-descriptor
distance between the reconstructed path and the target path.

KEY DESIGN CHOICE -- Magnitude-only FD distance
------------------------------------------------
The raw real+imag FD vector is phase-sensitive: it depends on which point of
the curve is treated as the "start".  The neural-network target FD starts at
whichever pixel OpenCV finds first; the FK trajectory starts at theta2=0.
Even for identical shapes these phases differ wildly, so minimising the raw
L2 distance would chase phase alignment, not shape matching.

We therefore compare MAGNITUDES |F_k| only, which are invariant to the
traversal starting point and give a true shape-similarity metric.

Public API
----------
    refine_params(fd_target, init_params, n_steps, n_restarts)
        -> dict  (same keys as predict_linkage output, plus 'refined': True)
"""

import numpy as np
from scipy.optimize import minimize

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_generation import (
    is_crank_rocker,
    forward_kinematics_trajectory,
    compute_fourier_descriptors,
)

NUM_FOURIER     = 15
NUM_TRAJ_POINTS = 256   # higher resolution for FD computation
_PARAM_KEYS     = ["norm_a", "norm_b", "norm_c", "norm_d", "norm_p", "p_angle"]

# ── Encoding helpers ───────────────────────────────────────────────────────────
# We optimise in unconstrained space and decode to bounded params.

def _encode(params: dict) -> np.ndarray:
    """Convert params dict -> unconstrained array via logit for ratios."""
    def logit(x, lo=0.05, hi=1.0):
        x = np.clip(x, lo + 1e-6, hi - 1e-6)
        p = (x - lo) / (hi - lo)
        return np.log(p / (1 - p))

    return np.array([
        logit(params["norm_a"]),
        logit(params["norm_b"]),
        logit(params["norm_c"]),
        logit(params["norm_d"]),
        logit(params["norm_p"]),
        params["p_angle"],          # already unbounded
    ])


def _decode(x: np.ndarray) -> dict:
    """Convert unconstrained array -> params dict via sigmoid for ratios."""
    def sigmoid(z, lo=0.05, hi=1.0):
        return lo + (hi - lo) / (1 + np.exp(-z))

    return {
        "norm_a":  sigmoid(x[0]),
        "norm_b":  sigmoid(x[1]),
        "norm_c":  sigmoid(x[2]),
        "norm_d":  sigmoid(x[3]),
        "norm_p":  sigmoid(x[4]),
        "p_angle": float(np.arctan2(np.sin(x[5]), np.cos(x[5]))),
    }


def _random_crank_rocker_params(rng) -> dict:
    """
    Sample a random valid crank-rocker configuration.
    a is the shortest link (crank), d=1 fixed (ground).
    Returns normalised params (same schema as model output).
    """
    for _ in range(500):
        # Sample d=1 already set, draw a, b, c, p_dist uniformly
        a  = rng.uniform(0.05, 0.30)   # crank: kept short so it's the shortest
        b  = rng.uniform(a + 0.01, 0.95)
        c  = rng.uniform(a + 0.01, 0.95)
        d  = 1.0
        pp = rng.uniform(0.05, 1.0)
        phi = rng.uniform(-np.pi, np.pi)
        if is_crank_rocker(a, b, c, d):
            return {
                "norm_a": a,  "norm_b": b,
                "norm_c": c,  "norm_d": d,
                "norm_p": pp, "p_angle": phi,
            }
    return None


# ── Objective function ─────────────────────────────────────────────────────────

def _fd_magnitudes(fd_vec: np.ndarray) -> np.ndarray:
    """
    Convert interleaved [Re(F1), Im(F1), Re(F2), Im(F2), ...] to magnitudes
    [|F1|, |F2|, ...].  This is invariant to the starting point of the curve.
    """
    re = fd_vec[0::2]
    im = fd_vec[1::2]
    return np.sqrt(re**2 + im**2)


def _fd_distance(fd_pred: np.ndarray, fd_tgt_mag: np.ndarray) -> float:
    """
    Magnitude-only FD distance.
    fd_tgt_mag should already be the pre-computed magnitude vector of the target.
    """
    return float(np.linalg.norm(_fd_magnitudes(fd_pred) - fd_tgt_mag))


_MAX_LINK_RATIO  = 3.0   # maximum ratio between any two link lengths
_AR_WEIGHT       = 3.0   # penalty weight for aspect-ratio mismatch


def _aspect_ratio(Px, Py) -> float:
    """
    Height/width ratio of the path's bounding box.
    Rotation-invariant (uses PCA-aligned bounding box).
    Value is always >= 1 (larger axis over smaller axis).
    """
    # PCA to find principal axes (avoids dependency on orientation)
    pts = np.column_stack([Px - Px.mean(), Py - Py.mean()])
    cov = pts.T @ pts / len(pts)
    vals, vecs = np.linalg.eigh(cov)   # ascending eigenvalues
    # Project onto principal axes
    proj = pts @ vecs
    extents = proj.max(axis=0) - proj.min(axis=0)
    ax1, ax2 = extents[0] + 1e-9, extents[1] + 1e-9
    return max(ax1, ax2) / min(ax1, ax2)


def _objective(x: np.ndarray, fd_tgt_mag: np.ndarray,
               tgt_ar: float) -> float:
    """
    Fast combined objective:
      - Magnitude FD distance  (phase-invariant shape spectrum)
      - Aspect-ratio penalty   (catches degenerate open-arc solutions cheaply)
      - Link-ratio penalty     (prevents extreme degenerate link lengths)

    The aspect ratio uses a PCA-aligned bounding box, so it is rotation-invariant.
    This is O(N) and runs at full Nelder-Mead speed (unlike Chamfer which requires
    36 KDTree queries per evaluation step).
    """
    p = _decode(x)

    nd = p["norm_d"]
    if nd < 1e-6:
        return 1e6

    a  = p["norm_a"] / nd
    b  = p["norm_b"] / nd
    c  = p["norm_c"] / nd
    d  = 1.0
    pp = p["norm_p"] / nd
    phi = p["p_angle"]

    # Enforce crank-rocker
    if not is_crank_rocker(a, b, c, d):
        return 1e6

    # Soft penalty: extreme link ratios lead to degenerate open-arc paths
    all_links = [a, b, c, d, pp]
    shortest  = min(all_links)
    if shortest < 1e-9:
        return 1e6
    ratio_penalty = 0.0
    for L in all_links:
        excess = L / shortest - _MAX_LINK_RATIO
        if excess > 0:
            ratio_penalty += excess**2

    res = forward_kinematics_trajectory(a, b, c, d, pp, phi,
                                        num_points=NUM_TRAJ_POINTS)
    if res is None:
        return 1e6

    Px, Py = res
    if np.any(np.isnan(Px)) or np.any(np.isnan(Py)):
        return 1e6

    span = max(Px.max() - Px.min(), Py.max() - Py.min())
    if span < 1e-4:
        return 1e6

    # Aspect ratio penalty (rotation-invariant, very fast)
    pred_ar = _aspect_ratio(Px, Py)
    ar_loss  = (np.log(pred_ar) - np.log(tgt_ar)) ** 2   # log-ratio is symmetric

    fd_pred = compute_fourier_descriptors(Px, Py, num_descriptors=NUM_FOURIER)
    fd_loss = _fd_distance(fd_pred, fd_tgt_mag)

    return fd_loss + _AR_WEIGHT * ar_loss + 0.5 * ratio_penalty


# ── Single-run Nelder-Mead ─────────────────────────────────────────────────────

def _run_nelder_mead(x0: np.ndarray, fd_tgt_mag: np.ndarray,
                     tgt_ar: float,
                     n_steps: int) -> tuple:
    """Returns (best_x, best_loss, n_iters)."""
    init_loss = _objective(x0, fd_tgt_mag, tgt_ar)
    if init_loss >= 1e5:          # invalid starting point
        return x0, init_loss, 0

    result = minimize(
        _objective,
        x0,
        args=(fd_tgt_mag, tgt_ar),
        method="Nelder-Mead",
        options={
            "maxiter":  n_steps,
            "xatol":    1e-5,
            "fatol":    1e-5,
            "adaptive": True,
        },
    )
    final_loss = result.fun if result.fun < 1e5 else init_loss
    return result.x, final_loss, result.nit


# ── Public API ─────────────────────────────────────────────────────────────────

def refine_params(
    fd_target:   np.ndarray,
    Px_target:   np.ndarray,
    Py_target:   np.ndarray,
    init_params: dict,
    n_steps:     int  = 2000,
    n_restarts:  int  = 5,
    verbose:     bool = False,
) -> dict:
    """
    Refine the initial model prediction by minimising FD-magnitude + aspect-ratio.

    Parameters
    ----------
    fd_target   : [30] raw (scale-normalised) FD vector from the input image.
    Px_target   : raw x pixel coordinates of input contour (for aspect ratio).
    Py_target   : raw y pixel coordinates of input contour (for aspect ratio).
    init_params : dict with keys norm_a..norm_p, p_angle  (from predict_linkage).
    n_steps     : max function evaluations per Nelder-Mead run.
    n_restarts  : number of random restarts in addition to the model prediction.
    verbose     : print progress.
    """
    # Pre-compute once: target FD magnitudes and aspect ratio
    fd_tgt_mag = _fd_magnitudes(fd_target)
    tgt_ar     = _aspect_ratio(Px_target, Py_target)

    if verbose:
        print(f"  [refine] Target aspect ratio = {tgt_ar:.3f}")

    # Initial combined loss from model prediction
    x0_model  = _encode(init_params)
    init_loss = _objective(x0_model, fd_tgt_mag, tgt_ar)

    if verbose:
        print(f"  [refine] Initial combined loss = {init_loss:.4f}")

    best_x    = x0_model
    best_loss = init_loss
    best_nit  = 0

    # 1) Refine from model's starting point
    x_opt, loss_opt, nit = _run_nelder_mead(x0_model, fd_tgt_mag, tgt_ar, n_steps)
    if loss_opt < best_loss:
        best_x, best_loss, best_nit = x_opt, loss_opt, nit

    # 2) Random restarts
    rng = np.random.default_rng(seed=42)
    for i in range(n_restarts):
        rp = _random_crank_rocker_params(rng)
        if rp is None:
            continue
        x0_r = _encode(rp)
        x_r, loss_r, nit_r = _run_nelder_mead(x0_r, fd_tgt_mag, tgt_ar, n_steps)
        if verbose:
            init_r = _objective(x0_r, fd_tgt_mag, tgt_ar)
            print(f"  [refine] Restart {i+1}/{n_restarts}: "
                  f"init={init_r:.4f} -> {loss_r:.4f}")
        if loss_r < best_loss:
            best_x, best_loss, best_nit = x_r, loss_r, nit_r

    improvement = max(0.0, (init_loss - best_loss) / (init_loss + 1e-9) * 100)

    if verbose:
        print(f"  [refine] Final combined loss  = {best_loss:.4f}  "
              f"({improvement:.1f}% improvement, {best_nit} iters)")

    refined = _decode(best_x)
    refined["refined"]          = True
    refined["init_fd_dist"]     = float(init_loss)
    refined["refined_fd_dist"]  = float(best_loss)
    refined["improvement_pct"]  = float(improvement)

    return refined
