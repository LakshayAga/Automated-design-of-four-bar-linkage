"""
run_inference.py -- End-to-end inference for a hand-drawn linkage path image.

HOW TO USE
----------
1. Drop your drawing image (PNG or JPG) into the  inputs/  folder.
2. Run from the project root:

       python scripts/run_inference.py                  # auto-picks the image in inputs/
       python scripts/run_inference.py my_drawing.png   # just the filename, looks in inputs/
       python scripts/run_inference.py path/to/any.png  # or give any full/relative path

3. Find all results in  runs/<image_name>/
       params.json            -- model-only predicted linkage parameters
       params_refined.json    -- after local optimisation refinement
       fd_diagnostic.png      -- 4-panel image-parser diagnostic
       reconstruction.png     -- 3-way comparison: input / model / refined
       summary.txt            -- human-readable report

Optional flags:
    --model  path/to/checkpoint.pth   use a different model (default: models/best_model.pth)
    --no-refine                       skip local optimisation (faster, model prediction only)
    --refine-steps N                  max Nelder-Mead evaluations (default: 3000)
    --show                            open plots on screen as well as saving them
"""

import os
import sys
import json
import argparse
import glob
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUTS_DIR   = os.path.join(PROJECT_ROOT, "inputs")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from predict    import predict_linkage, DEFAULT_MODEL
from data_generation import compute_fourier_descriptors
from reconstruct import reconstruct_fk, center_scale, check_params_validity
from inspect_image import extract_contour, extract_contour_raw, build_figure as _build_fd_figure
from refine     import refine_params

ACCENT  = "#2563eb"
GREEN   = "#16a34a"
ORANGE  = "#ea580c"
RED     = "#dc2626"
BG      = "#f8fafc"
DARK    = "#1e293b"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": "#cbd5e1", "axes.labelcolor": DARK,
    "xtick.color": DARK, "ytick.color": DARK,
    "text.color": DARK, "grid.color": "#e2e8f0",
    "font.family": "sans-serif",
})

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")


# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_image_path(arg):
    if arg is None:
        candidates = [
            f for f in glob.glob(os.path.join(INPUTS_DIR, "*"))
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        ]
        if not candidates:
            print(
                f"[ERROR] No image found in  inputs/  and no path was given.\n"
                f"        Drop a PNG/JPG into the inputs/ folder, or pass a path:\n"
                f"            python scripts/run_inference.py my_drawing.png"
            )
            sys.exit(1)
        if len(candidates) > 1:
            names = "\n          ".join(os.path.basename(c) for c in sorted(candidates))
            print(
                f"[ERROR] Multiple images in inputs/ -- please specify one:\n"
                f"          {names}\n"
                f"        Example:  python scripts/run_inference.py my_drawing.png"
            )
            sys.exit(1)
        return os.path.abspath(candidates[0])

    candidate = os.path.abspath(arg)
    if os.path.isfile(candidate):
        return candidate

    for ext in ("", *IMAGE_EXTS):
        p = os.path.join(INPUTS_DIR, arg + ext)
        if os.path.isfile(p):
            return os.path.abspath(p)

    print(
        f"[ERROR] Image not found: '{arg}'\n"
        f"        Tried as full path and in the inputs/ folder."
    )
    sys.exit(1)


def _best_rotation_align(Px_src, Py_src, Px_ref, Py_ref) -> tuple:
    """
    Find the 2-D rotation angle theta that best aligns (Px_src, Py_src) to
    (Px_ref, Py_ref) by minimising the sum of nearest-neighbour squared distances.
    Both curves must already be centred and unit-scaled.

    Because a four-bar linkage can be physically mounted at any orientation, the
    rotation of the coupler curve is a free parameter.  Aligning before plotting
    gives a fair visual comparison of SHAPE similarity.

    Algorithm: try a grid of angles and refine the best with scipy.minimize_scalar.
    """
    from scipy.optimize import minimize_scalar
    from scipy.spatial import cKDTree

    ref_tree = cKDTree(np.column_stack([Px_ref, Py_ref]))

    def cost(theta):
        c, s = np.cos(theta), np.sin(theta)
        rx = c * Px_src - s * Py_src
        ry = s * Px_src + c * Py_src
        dists, _ = ref_tree.query(np.column_stack([rx, ry]))
        return np.sum(dists**2)

    # Coarse grid search over [0, 2pi)
    thetas = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    costs  = [cost(t) for t in thetas]
    best_t = thetas[int(np.argmin(costs))]

    # Fine local refinement around best coarse angle
    res = minimize_scalar(cost, bounds=(best_t - 0.1, best_t + 0.1),
                          method="bounded")
    theta_opt = res.x
    c, s = np.cos(theta_opt), np.sin(theta_opt)
    return (c * Px_src - s * Py_src,
            s * Px_src + c * Py_src)


def _plot_reconstruction(Px_in, Py_in,
                         recon_model, reason_model,
                         recon_refined, reason_refined,
                         image_stem, run_dir, show):
    """
    Three-panel reconstruction comparison:
      Left  -- model prediction only
      Right -- after local optimisation refinement

    The reconstructed path is rotation-aligned to the input before plotting
    because the physical mounting orientation of a four-bar linkage is a free
    parameter (the coupler curve shape, not its orientation, is what matters).
    """
    has_model   = recon_model   is not None
    has_refined = recon_refined is not None

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor=BG)
    fig.suptitle(
        f"Path Reconstruction  ·  {image_stem}",
        fontsize=13, fontweight="bold", color=DARK, y=1.01
    )

    iPx, iPy = center_scale(Px_in.copy(), Py_in.copy())

    for ax, recon, reason, has, title, color in [
        (axes[0], recon_model,   reason_model,   has_model,   "Model Prediction",          ACCENT),
        (axes[1], recon_refined, reason_refined, has_refined, "After Local Optimisation",  ORANGE),
    ]:
        ax.set_facecolor(BG)
        ax.plot(iPx, iPy, color=DARK, linewidth=2.0, label="Input path", zorder=3, alpha=0.8)
        ax.plot(iPx[0], iPy[0], "o", color=DARK, markersize=8, zorder=5)

        if has:
            rPx, rPy = center_scale(recon[0].copy(), recon[1].copy())
            # Rotate reconstructed path to best match input (orientation is a free parameter)
            rPx, rPy = _best_rotation_align(rPx, rPy, iPx, iPy)
            ax.plot(rPx, rPy, color=color, linewidth=2.5, linestyle="--",
                    label="Reconstructed (rot-aligned)", zorder=4)
            ax.plot(rPx[0], rPy[0], "s", color=color, markersize=9, zorder=6)
            subtitle = "OK"
        else:
            ax.text(0, 0, f"Failed:\n{reason}",
                    ha="center", va="center", color=RED, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.6", facecolor="white", alpha=0.9))
            subtitle = f"FAILED"

        ax.set_title(f"{title}\n({subtitle})", fontsize=11,
                     color=DARK, fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=9)
        ax.set_xlabel("Normalised X")
        ax.set_ylabel("Normalised Y")

    fig.tight_layout()
    out = os.path.join(run_dir, "reconstruction.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"    Saved: {out}")
    return has_model, has_refined


def _plot_fd_diagnostic(image_path, Px_in, Py_in, img_gray, thresh, contours, run_dir, show):
    fd_vec = compute_fourier_descriptors(Px_in, Py_in, num_descriptors=15)
    fig    = _build_fd_figure(img_gray, thresh, Px_in, Py_in, contours, fd_vec, 15, image_path)
    out    = os.path.join(run_dir, "fd_diagnostic.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"    Saved: {out}")


def _save_params(p: dict, path: str, extra: dict = None):
    save = {k: v for k, v in p.items() if k not in ("fd_tensor", "fd_tensor_norm")}
    save["p_angle_deg"] = float(np.degrees(p["p_angle"]))
    if extra:
        save.update(extra)
    with open(path, "w") as f:
        json.dump(save, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Four-bar linkage predictor -- single image inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image",          nargs="?", default=None,
                        help="Filename in inputs/ or full path. Omit to auto-detect.")
    parser.add_argument("--model",        default=DEFAULT_MODEL)
    parser.add_argument("--no-refine",    action="store_true",
                        help="Skip local optimisation (faster).")
    parser.add_argument("--refine-steps", type=int, default=2000,
                        help="Max Nelder-Mead evaluations per run (default 2000).")
    parser.add_argument("--restarts",     type=int, default=5,
                        help="Random restarts for refinement (default 5).")
    parser.add_argument("--show",         action="store_true")
    args = parser.parse_args()

    image_path = resolve_image_path(args.image)
    if args.show:
        matplotlib.use("TkAgg")

    image_stem = os.path.splitext(os.path.basename(image_path))[0]
    run_dir    = os.path.join(PROJECT_ROOT, "runs", image_stem)
    os.makedirs(run_dir, exist_ok=True)

    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  Four-Bar Linkage Predictor  ·  Inference")
    print(f"  Image  : {image_path}")
    print(f"  Model  : {args.model}")
    print(f"  Output : {run_dir}")
    print(f"  Refine : {'disabled (--no-refine)' if args.no_refine else f'Nelder-Mead ({args.refine_steps} steps, {args.restarts} restarts)'}")
    print(f"{sep}\n")

    # ── 1. Extract paths ──────────────────────────────────────────────────────
    print("[1/5] Extracting path from image ...")
    # RAW pixels -> used for FD computation and model (preserves training distribution)
    Px_raw, Py_raw = extract_contour_raw(image_path)
    # Smoothed display path -> used for visualisation only (no overlap artefacts)
    img_gray, thresh, Px_disp, Py_disp, contours = extract_contour(image_path)
    fd_target = compute_fourier_descriptors(Px_raw, Py_raw, num_descriptors=15)
    print(f"    Raw contour  : {len(Px_raw)} pts  |  FD vector: {fd_target.shape}")
    print(f"    Display path : {len(Px_disp)} pts (smoothed, for visualisation only)")


    # ── 2. Model prediction ───────────────────────────────────────────────────
    print("\n[2/5] Running neural network prediction ...")
    t0     = time.time()
    params = predict_linkage(image_path, model_path=args.model)
    t_pred = time.time() - t0

    init_keys = ["norm_a", "norm_b", "norm_c", "norm_d", "norm_p", "p_angle"]
    fk_input  = {k: params[k] for k in init_keys}

    print(f"\n  Model prediction ({t_pred*1000:.0f} ms):")
    for k in ["norm_a", "norm_b", "norm_c", "norm_d", "norm_p"]:
        print(f"    {k:<8} = {params[k]:+.4f}")
    print(f"    p_angle  = {params['p_angle']:+.4f} rad  ({np.degrees(params['p_angle']):+.1f}°)")

    valid_m, reason_m = check_params_validity(fk_input)
    print(f"    Valid    = {valid_m}  ({reason_m})")

    _save_params(params, os.path.join(run_dir, "params.json"),
                 extra={"params_valid": valid_m, "validity_note": reason_m,
                        "stage": "model_prediction"})
    print(f"    Saved: params.json")

    # ── 3. Local optimisation refinement ──────────────────────────────────────
    refined_params = None
    if not args.no_refine:
        print(f"\n[3/5] Local optimisation (Chamfer+FD, {args.refine_steps} steps x {args.restarts+1} runs) ...")
        t0 = time.time()
        refined_params = refine_params(
            fd_target   = fd_target,
            Px_target   = Px_raw,
            Py_target   = Py_raw,
            init_params = fk_input,
            n_steps     = args.refine_steps,
            n_restarts  = args.restarts,
            verbose     = True,
        )
        t_refine = time.time() - t0

        print(f"\n  Refined parameters ({t_refine:.1f} s):")
        for k in ["norm_a", "norm_b", "norm_c", "norm_d", "norm_p"]:
            model_val = params[k]
            ref_val   = refined_params[k]
            delta     = ref_val - model_val
            print(f"    {k:<8} = {ref_val:+.4f}  (D={delta:+.4f})")
        print(f"    p_angle  = {refined_params['p_angle']:+.4f} rad  ({np.degrees(refined_params['p_angle']):+.1f} deg)")
        print(f"    FD dist  : {refined_params['init_fd_dist']:.4f} -> {refined_params['refined_fd_dist']:.4f}  "
              f"({refined_params['improvement_pct']:.1f}% improvement)")

        valid_r, reason_r = check_params_validity(refined_params)
        print(f"    Valid    = {valid_r}  ({reason_r})")

        _save_params(refined_params, os.path.join(run_dir, "params_refined.json"),
                     extra={"params_valid": valid_r, "validity_note": reason_r,
                            "stage": "refined"})
        print(f"    Saved: params_refined.json")
    else:
        print("\n[3/5] Skipping local optimisation (--no-refine).")
        valid_r, reason_r = False, "skipped"

    # ── 4. FD diagnostic ──────────────────────────────────────────────────────
    print("\n[4/5] Generating FD diagnostic figure ...")
    _plot_fd_diagnostic(image_path, Px_disp, Py_disp, img_gray, thresh, contours,
                        run_dir, args.show)

    # ── 5. Reconstruction plot ────────────────────────────────────────────────
    print("\n[5/5] Generating reconstruction plot ...")
    recon_m,  reason_fk_m  = reconstruct_fk(fk_input)
    recon_r,  reason_fk_r  = (
        reconstruct_fk({k: refined_params[k] for k in init_keys})
        if refined_params is not None
        else (None, "refinement skipped")
    )
    if recon_m  is None: print(f"  [WARN] Model    FK: {reason_fk_m}")
    if recon_r  is None: print(f"  [WARN] Refined  FK: {reason_fk_r}")

    _plot_reconstruction(
        Px_disp, Py_disp,
        recon_m  if recon_m  is not None else None,  reason_fk_m,
        recon_r  if recon_r  is not None else None,  reason_fk_r,
        image_stem, run_dir, args.show
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    lines = [
        "Four-Bar Linkage Inference Summary",
        "=" * 40,
        f"Image      : {image_path}",
        f"Model      : {args.model}",
        f"Run dir    : {run_dir}",
        "",
        "Model Prediction:",
        f"  Valid    : {valid_m}  ({reason_m})",
    ]
    for k in init_keys:
        lines.append(f"  {k:<8} : {params[k]:.6f}")
    if refined_params:
        lines += [
            "",
            "After Refinement:",
            f"  Valid    : {valid_r}  ({reason_r})",
            f"  FD dist  : {refined_params['init_fd_dist']:.4f} -> {refined_params['refined_fd_dist']:.4f}"
            f"  ({refined_params['improvement_pct']:.1f}% better)",
        ]
        for k in init_keys:
            lines.append(f"  {k:<8} : {refined_params[k]:.6f}")
    lines += [
        "",
        "Output files:",
        "  params.json          model-only parameters",
        "  params_refined.json  parameters after Nelder-Mead optimisation",
        "  fd_diagnostic.png    image-parser diagnostic",
        "  reconstruction.png   model vs refined path comparison",
        "  summary.txt          this file",
    ]
    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines))

    print(f"\n{sep}")
    print(f"  Done!  Results -> runs/{image_stem}/")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
