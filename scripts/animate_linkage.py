"""
animate_linkage.py  --  Animate a four-bar linkage from a run folder.

HOW TO USE
----------
Run from the project root, passing the run folder created by run_inference.py:

    python scripts/animate_linkage.py runs/trace_trial1

    # Save a GIF instead of showing the window:
    python scripts/animate_linkage.py runs/trace_trial1 --save

    # Save an MP4 (requires ffmpeg on PATH):
    python scripts/animate_linkage.py runs/trace_trial1 --save --fmt mp4

    # Skip the input-trace overlay (linkage animation only):
    python scripts/animate_linkage.py runs/trace_trial1 --no-overlay

    # Control animation speed (default 120 frames, 30 ms/frame):
    python scripts/animate_linkage.py runs/trace_trial1 --frames 200 --interval 20

PARAMS RESOLUTION (inside the folder)
--------------------------------------
    1. params_refined.json   (preferred -- post-optimisation)
    2. params.json           (fallback  -- model-only prediction)

INPUT TRACE OVERLAY
-------------------
The run folder name is used to find the original drawing:
    runs/trace_trial1/  ->  inputs/trace_trial1.png

LAYOUT
------
  Left panel  : Physical mechanism -- pivots, links, rotating crank, growing coupler trace.
  Right panel : Normalised overlay  -- FK trace (growing, coloured) vs. input drawing (grey).

The coupler point P is shown as a filled circle and leaves a colour gradient trail.
The mechanism links are drawn as thick lines:
    O2 -- crank (a) --> B -- coupler (b) -- C -- rocker (c) -- O4
    O2 --------------- ground (d) ---------------------------- O4
"""

import os
import sys
import json
import argparse
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

# ── Project path setup ────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
INPUTS_DIR = os.path.join(PROJECT_ROOT, "inputs")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

# ── Colour palette (matches the rest of the project) ──────────────────────────
BG      = "#f8fafc"
DARK    = "#1e293b"
ACCENT  = "#2563eb"   # crank
ORANGE  = "#ea580c"   # coupler / trace
GREEN   = "#16a34a"   # rocker
PURPLE  = "#7c3aed"   # coupler point P
GREY    = "#94a3b8"   # ground link & input trace

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": "#cbd5e1", "axes.labelcolor": DARK,
    "xtick.color": DARK, "ytick.color": DARK,
    "text.color": DARK, "grid.color": "#e2e8f0",
    "font.family": "sans-serif",
})


# ── Forward kinematics (self-contained, no import needed) ─────────────────────

def _solve_fk(a, b, c, d, p_dist, p_angle, n_frames=120):
    """
    Compute all joint positions over one full crank revolution.

    Returns a list of dicts, one per frame:
        O2  : (0, 0)          crank pivot   (fixed)
        O4  : (d, 0)          rocker pivot  (fixed)
        B   : crank pin       (end of crank, start of coupler)
        C   : rocker pin      (end of coupler, end of rocker)
        P   : coupler point   (the end-effector)
    Returns None if the linkage is non-assembable for any frame.
    """
    theta2_vals = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    O2 = np.array([0.0, 0.0])
    O4 = np.array([d,   0.0])

    frames = []
    for t2 in theta2_vals:
        K1 = d / a
        K2 = d / c
        K3 = (a**2 - b**2 + c**2 + d**2) / (2.0 * a * c)

        A_ = np.cos(t2) - K1 - K2 * np.cos(t2) + K3
        B_ = -2.0 * np.sin(t2)
        C_ = K1 - (K2 + 1.0) * np.cos(t2) + K3

        disc = B_**2 - 4.0 * A_ * C_
        if disc < 0:
            return None          # linkage can't assemble at this angle

        t4 = 2.0 * np.arctan((-B_ - np.sqrt(disc)) / (2.0 * A_))
        s3 = (c * np.sin(t4) - a * np.sin(t2)) / b
        c3 = (d + c * np.cos(t4) - a * np.cos(t2)) / b
        t3 = np.arctan2(s3, c3)

        B_pt = O2 + a * np.array([np.cos(t2), np.sin(t2)])
        C_pt = O4 + c * np.array([np.cos(t4), np.sin(t4)])
        P_pt = B_pt + p_dist * np.array([np.cos(t3 + p_angle),
                                          np.sin(t3 + p_angle)])

        frames.append({"O2": O2, "O4": O4, "B": B_pt, "C": C_pt, "P": P_pt,
                        "t3": t3, "t2": t2})
    return frames


def _center_scale(Px, Py):
    """Translate to centroid, scale so max span = 1."""
    Px = Px - Px.mean()
    Py = Py - Py.mean()
    span = max(Px.max() - Px.min(), Py.max() - Py.min())
    if span > 1e-9:
        Px, Py = Px / span, Py / span
    return Px, Py


def _best_rotation_align(Px_src, Py_src, Px_ref, Py_ref):
    """Rotate (Px_src, Py_src) to best match (Px_ref, Py_ref) via coarse grid."""
    from scipy.spatial import cKDTree
    from scipy.optimize import minimize_scalar

    ref_tree = cKDTree(np.column_stack([Px_ref, Py_ref]))

    def cost(theta):
        co, si = np.cos(theta), np.sin(theta)
        rx = co * Px_src - si * Py_src
        ry = si * Px_src + co * Py_src
        dists, _ = ref_tree.query(np.column_stack([rx, ry]))
        return float(np.sum(dists**2))

    thetas = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    best_t = thetas[int(np.argmin([cost(t) for t in thetas]))]
    res    = minimize_scalar(cost, bounds=(best_t - 0.1, best_t + 0.1),
                             method="bounded")
    co, si = np.cos(res.x), np.sin(res.x)
    return co * Px_src - si * Py_src, si * Px_src + co * Py_src


# ── Smoothing helper (mirrors inspect_image.smooth_contour) ───────────────────

def _resample_arc(Px, Py, N=512):
    """Resample a closed curve to N arc-length-equidistant points."""
    from scipy.ndimage import uniform_filter1d  # already in scipy, always available
    cx = np.append(Px, Px[0])
    cy = np.append(Py, Py[0])
    ds  = np.sqrt(np.diff(cx)**2 + np.diff(cy)**2)
    arc = np.concatenate([[0], np.cumsum(ds)])
    t   = np.linspace(0, arc[-1], N, endpoint=False)
    return np.interp(t, arc, cx), np.interp(t, arc, cy)


def _smooth_contour(Px, Py, window=15, n=512):
    """Arc-resample then apply a light circular moving-average (matches inspect_image)."""
    from scipy.ndimage import uniform_filter1d
    Px_rs, Py_rs = _resample_arc(Px, Py, n)
    if window > 1:
        def cma(a):
            ext = np.concatenate([a[-window:], a, a[:window]])
            return uniform_filter1d(ext, size=window)[window:-window]
        Px_rs, Py_rs = cma(Px_rs), cma(Py_rs)
    return Px_rs, Py_rs


# ── Input-image overlay helper ────────────────────────────────────────────────

def _load_input_trace(params_json_path):
    """
    Find the input image for this run folder and return its smoothed contour.

    Layout:  runs/<stem>/params_refined.json  -->  inputs/<stem>.png

    Inlines the contour extraction + smoothing to avoid importing inspect_image
    (which pulls in cv2 at module level and raises ImportError if cv2 is not
    on the active Python path).

    Returns (Px, Py) in normalised Cartesian coords, or (None, None).
    """
    # Guard cv2 import -- give a clear message if missing
    try:
        import cv2
    except ImportError:
        print("  [WARN] opencv-python (cv2) not found -- overlay skipped.\n"
              "         Install it with:  pip install opencv-python-headless")
        return None, None

    run_dir = os.path.dirname(os.path.abspath(params_json_path))
    stem    = os.path.basename(run_dir)

    # Search inputs/ for a matching image file
    found_image = None
    for ext in IMAGE_EXTS:
        candidate = os.path.join(INPUTS_DIR, stem + ext)
        if os.path.isfile(candidate):
            found_image = candidate
            break

    if found_image is None:
        print(f"  [WARN] No matching input image found for '{stem}' in: {INPUTS_DIR}")
        return None, None

    print(f"  [overlay] Loading smoothed contour from: {os.path.basename(found_image)}")
    try:
        img = cv2.imread(found_image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"cv2 could not read: {found_image}")

        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)
        if not contours:
            raise ValueError("No contours found in image.")

        largest = max(contours, key=len).squeeze(1)
        if len(largest) < 10:
            raise ValueError(f"Contour too small ({len(largest)} pts).")

        Px_raw = largest[:, 0].astype(float)
        # Flip Y: image origin is top-left; convert to Cartesian (y up)
        Py_raw = (img.shape[0] - largest[:, 1]).astype(float)

        # Smooth: arc-resample + moving average (same as inspect_image.smooth_contour)
        Px_sm, Py_sm = _smooth_contour(Px_raw, Py_raw)

        # Normalise to [-0.5, 0.5] centroid-centred unit span
        Px_sm, Py_sm = _center_scale(Px_sm, Py_sm)

        print(f"  [overlay] Smoothed contour: {len(Px_sm)} pts")
        return Px_sm, Py_sm

    except Exception as e:
        print(f"  [WARN] Failed to extract contour: {e}")
        return None, None


# ── Load params ───────────────────────────────────────────────────────────────

def _load_params(path):
    """Read a params JSON and return physical link lengths (with d normalised to 1)."""
    with open(path) as f:
        p = json.load(f)
    nd = p["norm_d"]
    if nd < 1e-9:
        raise ValueError("norm_d is zero – invalid params file.")
    return {
        "a":       p["norm_a"] / nd,
        "b":       p["norm_b"] / nd,
        "c":       p["norm_c"] / nd,
        "d":       1.0,
        "p_dist":  p["norm_p"] / nd,
        "p_angle": p["p_angle"],
        # keep raw for display
        "norm_a":  p["norm_a"], "norm_b": p["norm_b"],
        "norm_c":  p["norm_c"], "norm_d": p["norm_d"],
        "norm_p":  p["norm_p"],
    }


# ── Main animation builder ────────────────────────────────────────────────────

def build_animation(params_json, n_frames=120, interval=30,
                    show_overlay=True, save=False, fmt="gif"):
    # ── 1. Load params & run FK ───────────────────────────────────────────────
    pm = _load_params(params_json)
    a, b, c, d = pm["a"], pm["b"], pm["c"], pm["d"]
    p_dist, p_angle = pm["p_dist"], pm["p_angle"]

    frames = _solve_fk(a, b, c, d, p_dist, p_angle, n_frames)
    if frames is None:
        print("[ERROR] Forward kinematics failed – linkage cannot assemble.")
        sys.exit(1)

    # Collect full coupler trace for the right panel
    Px_fk = np.array([f["P"][0] for f in frames])
    Py_fk = np.array([f["P"][1] for f in frames])
    Px_n,  Py_n  = _center_scale(Px_fk.copy(), Py_fk.copy())

    # ── 2. Optional input-trace overlay ──────────────────────────────────────
    Px_in, Py_in = None, None
    if show_overlay:
        Px_in, Py_in = _load_input_trace(params_json)
        if Px_in is not None:
            # Align FK trace to input trace rotation
            Px_n, Py_n = _best_rotation_align(Px_n, Py_n, Px_in, Py_in)

    # ── 3. Figure layout ──────────────────────────────────────────────────────
    fig, (ax_mech, ax_trace) = plt.subplots(
        1, 2, figsize=(14, 6.5), facecolor=BG,
        gridspec_kw={"width_ratios": [1.2, 1]}
    )
    stem = os.path.basename(os.path.dirname(os.path.abspath(params_json)))
    fig.suptitle(
        f"Four-Bar Linkage Animation  ·  {stem}",
        fontsize=13, fontweight="bold", color=DARK, y=1.01
    )

    # ── Left panel: mechanism ─────────────────────────────────────────────────
    ax_mech.set_facecolor(BG)
    ax_mech.set_aspect("equal")
    ax_mech.grid(True, linestyle="--", alpha=0.4)
    ax_mech.set_title("Mechanism", fontsize=11, color=DARK, fontweight="bold")

    # Compute axis limits from all joint positions
    all_x = np.concatenate([[f["O2"][0], f["O4"][0], f["B"][0],
                              f["C"][0],  f["P"][0]] for f in frames])
    all_y = np.concatenate([[f["O2"][1], f["O4"][1], f["B"][1],
                              f["C"][1],  f["P"][1]] for f in frames])
    pad = (max(all_x) - min(all_x)) * 0.15 + 0.1
    ax_mech.set_xlim(all_x.min() - pad, all_x.max() + pad)
    ax_mech.set_ylim(all_y.min() - pad, all_y.max() + pad)
    ax_mech.set_xlabel("X (d = 1.0)", fontsize=9)
    ax_mech.set_ylabel("Y", fontsize=9)

    # Fixed elements
    O2, O4 = frames[0]["O2"], frames[0]["O4"]
    ax_mech.plot([O2[0], O4[0]], [O2[1], O4[1]],
                 color=GREY, linewidth=3.5, zorder=1, solid_capstyle="round",
                 label=f"Ground  d={d:.3f}")
    # Ground hatch marks
    for i, pt in enumerate([O2, O4]):
        ax_mech.plot(*pt, "s", color=DARK, markersize=10, zorder=8)
        ax_mech.plot(*pt, "s", color=BG, markersize=6, zorder=9)
    # Ghost full coupler trace
    ax_mech.plot(Px_fk, Py_fk, color=PURPLE, linewidth=1.0,
                 linestyle="--", alpha=0.25, zorder=2)

    # Animated artists for mechanism
    ln_crank,  = ax_mech.plot([], [], color=ACCENT,  linewidth=4,
                               solid_capstyle="round", zorder=5, label=f"Crank  a={a:.3f}")
    ln_coupler,= ax_mech.plot([], [], color=ORANGE, linewidth=3,
                               solid_capstyle="round", zorder=4, label=f"Coupler b={b:.3f}")
    ln_rocker, = ax_mech.plot([], [], color=GREEN,  linewidth=4,
                               solid_capstyle="round", zorder=5, label=f"Rocker  c={c:.3f}")
    # Coupler triangle (B → P → C, shaded)
    tri_patch = plt.Polygon([[0,0],[0,0],[0,0]], closed=True,
                             facecolor=ORANGE, alpha=0.15, zorder=3)
    ax_mech.add_patch(tri_patch)
    # Joint markers
    dot_B, = ax_mech.plot([], [], "o", color=ACCENT,  markersize=9,  zorder=7)
    dot_C, = ax_mech.plot([], [], "o", color=GREEN,   markersize=9,  zorder=7)
    dot_P, = ax_mech.plot([], [], "o", color=PURPLE,  markersize=11, zorder=10)
    # Growing coupler trace in mechanism panel
    trace_m_x, trace_m_y = [], []
    ln_trace_m, = ax_mech.plot([], [], color=PURPLE, linewidth=1.8,
                                alpha=0.7, zorder=6)
    # Crank angle arc (tiny visual)
    angle_text = ax_mech.text(
        O2[0] + 0.07, O2[1] + 0.07, "", fontsize=8, color=ACCENT, zorder=11
    )

    ax_mech.legend(loc="upper right", fontsize=8, framealpha=0.8)

    # ── Right panel: normalised trace overlay ─────────────────────────────────
    ax_trace.set_facecolor(BG)
    ax_trace.set_aspect("equal")
    ax_trace.grid(True, linestyle="--", alpha=0.4)
    ax_trace.set_title("Coupler Trace  (normalised)", fontsize=11,
                        color=DARK, fontweight="bold")
    ax_trace.set_xlabel("Normalised X", fontsize=9)
    ax_trace.set_ylabel("Normalised Y", fontsize=9)

    # Ghost full FK trace
    ax_trace.plot(Px_n, Py_n, color=PURPLE, linewidth=1.0,
                  linestyle="--", alpha=0.2, zorder=1)

    # Input trace overlay
    if Px_in is not None:
        ax_trace.plot(Px_in, Py_in, color=DARK, linewidth=2.2,
                      alpha=0.7, zorder=2, label="Input drawing")
        ax_trace.plot(Px_in[0], Py_in[0], "o", color=DARK,
                      markersize=8, zorder=3)

    # Growing animated FK trace
    trace_t_x, trace_t_y = [], []
    ln_trace_t, = ax_trace.plot([], [], color=PURPLE, linewidth=2.5,
                                 zorder=5, label="FK trace (growing)")
    dot_Pt, = ax_trace.plot([], [], "o", color=PURPLE, markersize=10, zorder=6)
    frame_counter = ax_trace.text(
        0.02, 0.97, "", transform=ax_trace.transAxes,
        fontsize=9, color=DARK, va="top"
    )

    # Normalised dot for current FK point on input overlay
    dot_curr, = ax_trace.plot([], [], "D", color=ORANGE, markersize=8, zorder=7)

    ax_trace.legend(loc="lower right", fontsize=8, framealpha=0.8)

    # Set nice limits for right panel
    pad2 = 0.12
    if Px_in is not None:
        all_nx = np.concatenate([Px_n, Px_in])
        all_ny = np.concatenate([Py_n, Py_in])
    else:
        all_nx, all_ny = Px_n, Py_n
    ax_trace.set_xlim(all_nx.min() - pad2, all_nx.max() + pad2)
    ax_trace.set_ylim(all_ny.min() - pad2, all_ny.max() + pad2)

    fig.tight_layout()

    # ── 4. Animation update function ──────────────────────────────────────────
    def update(i):
        f = frames[i]
        B, C, P = f["B"], f["C"], f["P"]

        # Mechanism links
        ln_crank.set_data([O2[0], B[0]], [O2[1], B[1]])
        ln_coupler.set_data([B[0], C[0]], [B[1], C[1]])
        ln_rocker.set_data([O4[0], C[0]], [O4[1], C[1]])

        # Coupler triangle (semi-transparent)
        tri_patch.set_xy([B, P, C])

        # Joint dots
        dot_B.set_data([B[0]], [B[1]])
        dot_C.set_data([C[0]], [C[1]])
        dot_P.set_data([P[0]], [P[1]])

        # Growing mechanism trace
        trace_m_x.append(P[0]);  trace_m_y.append(P[1])
        ln_trace_m.set_data(trace_m_x, trace_m_y)

        # Crank angle label
        deg = np.degrees(f["t2"])
        angle_text.set_text(f"θ₂ = {deg:.0f}°")

        # Normalised trace (right panel)
        trace_t_x.append(Px_n[i]);  trace_t_y.append(Py_n[i])
        ln_trace_t.set_data(trace_t_x, trace_t_y)
        dot_Pt.set_data([Px_n[i]], [Py_n[i]])
        dot_curr.set_data([Px_n[i]], [Py_n[i]])

        pct = 100 * (i + 1) / n_frames
        frame_counter.set_text(f"Frame {i+1}/{n_frames}  ({pct:.0f}%)")

        return (ln_crank, ln_coupler, ln_rocker, tri_patch,
                dot_B, dot_C, dot_P, ln_trace_m, angle_text,
                ln_trace_t, dot_Pt, dot_curr, frame_counter)

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=interval,
        blit=False, repeat=True
    )

    # ── 5. Output ─────────────────────────────────────────────────────────────
    if save:
        out_dir  = os.path.dirname(os.path.abspath(params_json))
        out_name = os.path.join(out_dir, f"animation.{fmt}")
        print(f"  Saving animation to: {out_name}  (this may take a moment...)")
        if fmt == "gif":
            ani.save(out_name, writer="pillow", fps=1000 // interval)
        elif fmt == "mp4":
            ani.save(out_name, writer="ffmpeg",
                     fps=1000 // interval, dpi=120,
                     extra_args=["-vcodec", "libx264"])
        else:
            print(f"[WARN] Unknown format '{fmt}', defaulting to gif.")
            ani.save(out_name.replace(fmt, "gif"), writer="pillow",
                     fps=1000 // interval)
        print(f"  Done: {out_name}")
    else:
        plt.show()

    return ani


# ── Resolve run folder -> params JSON ────────────────────────────────────────

def _resolve_params(run_folder):
    """
    Given a run folder, return the path to the best available params JSON:
      1. params_refined.json  (preferred -- after local optimisation)
      2. params.json          (fallback  -- model-only prediction)
    Raises FileNotFoundError if neither exists.
    """
    for name in ("params_refined.json", "params.json"):
        candidate = os.path.join(run_folder, name)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"No params JSON found in '{run_folder}'.\n"
        f"Expected 'params_refined.json' or 'params.json' inside that folder."
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Animate a four-bar linkage from a run folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "run_folder",
        help="Path to a run folder created by run_inference.py "
             "(e.g. runs/trace_trial1). "
             "Uses params_refined.json if present, otherwise params.json."
    )
    parser.add_argument(
        "--frames", type=int, default=120,
        help="Number of animation frames (crank positions). Default: 120."
    )
    parser.add_argument(
        "--interval", type=int, default=30,
        help="Delay between frames in milliseconds. Default: 30."
    )
    parser.add_argument(
        "--no-overlay", action="store_true",
        help="Skip loading the input drawing overlay."
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save the animation to disk instead of opening a window."
    )
    parser.add_argument(
        "--fmt", choices=["gif", "mp4"], default="gif",
        help="Output format when --save is used. Default: gif."
    )
    args = parser.parse_args()

    # ── Validate run folder ───────────────────────────────────────────────────
    run_folder = os.path.abspath(args.run_folder)
    if not os.path.isdir(run_folder):
        print(f"[ERROR] Not a directory: {run_folder}")
        sys.exit(1)

    try:
        params_json = _resolve_params(run_folder)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    label = "refined" if "refined" in os.path.basename(params_json) else "model-only"
    print(f"\n  Run folder : {run_folder}")
    print(f"  Params     : {os.path.basename(params_json)}  ({label})")
    pm = _load_params(params_json)
    print(f"  Links (d=1): a={pm['a']:.4f}  b={pm['b']:.4f}  "
          f"c={pm['c']:.4f}  d={pm['d']:.4f}")
    print(f"  Coupler pt : p={pm['p_dist']:.4f}  "
          f"phi={np.degrees(pm['p_angle']):.1f} deg")
    print(f"  Animation  : {args.frames} frames @ {args.interval} ms/frame  "
          f"| overlay={'off' if args.no_overlay else 'on'}\n")

    # Need interactive backend when not saving
    if not args.save:
        import matplotlib
        matplotlib.use("TkAgg")   # switch from Agg so plt.show() opens a window
        import matplotlib.pyplot as plt  # noqa: F811

    build_animation(
        params_json  = params_json,
        n_frames     = args.frames,
        interval     = args.interval,
        show_overlay = not args.no_overlay,
        save         = args.save,
        fmt          = args.fmt,
    )


if __name__ == "__main__":
    main()
