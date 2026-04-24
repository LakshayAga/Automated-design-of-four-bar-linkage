"""
Generate full CRANK-ROCKER train / val / test datasets.

Key changes vs previous version:
  - Filters for CRANK-ROCKER mechanisms only (a = shortest link, Grashof holds)
  - Fourier descriptor features are z-score normalised (mean/std from training set),
    then clipped to [-5, 5] so tanh activations never saturate on outlier FD values.
    Val/test are normalised using the TRAINING set stats (no data leakage).

Usage (from project root):
    python src/generate_full_dataset.py
    python src/generate_full_dataset.py --train 75000 --val 15000 --test 3000 --force

Outputs:
    data/train_dataset.pt
    data/val_dataset.pt
    data/test_dataset.pt
    data/feature_norm.pt   <- mean/std from training set, used by predict.py

Each .pt file dict keys:
    features        [N, 30]  z-score normalised FD, clipped to [-5, 5]
    targets         [N, 6]   [norm_a..norm_p, p_angle]
    trajectories_x  [N, 128]
    trajectories_y  [N, 128]
    + metadata
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_generation import (
    is_crank_rocker,
    forward_kinematics_trajectory,
    compute_fourier_descriptors,
)

PROJECT_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR        = os.path.join(PROJECT_ROOT, "data")
NUM_FOURIER     = 15
NUM_TRAJ_POINTS = 128


def generate_split(num_samples: int, seed: int, split_name: str) -> dict:
    """Generate one dataset split (raw, before normalisation)."""
    np.random.seed(seed)

    fd_list, params_list, trajs_x, trajs_y = [], [], [], []
    pbar = tqdm(total=num_samples, desc=f"  {split_name:<6}", unit="sample")

    while len(fd_list) < num_samples:
        a, b, c, d = np.random.uniform(0.5, 10.0, size=4)
        p_dist     = np.random.uniform(0.5, 10.0)
        p_angle    = np.random.uniform(-np.pi, np.pi)

        # ---- CRANK-ROCKER filter (replaces plain Grashof check) ----
        if not is_crank_rocker(a, b, c, d):
            continue

        res = forward_kinematics_trajectory(
            a, b, c, d, p_dist, p_angle, num_points=NUM_TRAJ_POINTS
        )
        if res is None:
            continue

        Px, Py = res
        if np.any(np.isnan(Px)) or np.any(np.isnan(Py)):
            continue

        fd      = compute_fourier_descriptors(Px, Py, num_descriptors=NUM_FOURIER)
        max_len = max(a, b, c, d)
        target  = np.array([
            a / max_len, b / max_len, c / max_len,
            d / max_len, p_dist / max_len, p_angle,
        ])

        fd_list.append(fd)
        params_list.append(target)
        trajs_x.append(Px)
        trajs_y.append(Py)
        pbar.update(1)

    pbar.close()

    return {
        "features_raw": np.array(fd_list, dtype=np.float32),
        "targets":      np.array(params_list, dtype=np.float32),
        "trajs_x":      np.array(trajs_x, dtype=np.float32),
        "trajs_y":      np.array(trajs_y, dtype=np.float32),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate crank-rocker train/val/test datasets with normalised FD inputs."
    )
    parser.add_argument("--train", type=int, default=75000)
    parser.add_argument("--val",   type=int, default=15000)
    parser.add_argument("--test",  type=int, default=3000)
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    splits = [
        ("train", args.train, 42, "train_dataset.pt"),
        ("val",   args.val,   43, "val_dataset.pt"),
        ("test",  args.test,  44, "test_dataset.pt"),
    ]

    # Check if we can skip everything
    all_exist = all(
        os.path.exists(os.path.join(DATA_DIR, fn)) for _, _, _, fn in splits
    )
    if all_exist and not args.force:
        print("All dataset files already exist. Pass --force to regenerate.")
        return

    raw = {}
    for split_name, n_samples, seed, filename in splits:
        out_path = os.path.join(DATA_DIR, filename)
        if os.path.exists(out_path) and not args.force:
            print(f"  [SKIP] {filename} already exists.")
            continue

        print(f"\n{'='*60}")
        print(f"  {split_name.upper():<6} | n={n_samples} | seed={seed} | crank-rocker only")
        print(f"{'='*60}")
        raw[split_name] = generate_split(n_samples, seed, split_name)

    # ── Compute normalisation stats on TRAINING set only ──────────────────────
    norm_path = os.path.join(DATA_DIR, "feature_norm.pt")

    if "train" in raw:
        fd_train  = raw["train"]["features_raw"]
        fd_mean   = fd_train.mean(axis=0)
        fd_std    = fd_train.std(axis=0)
        fd_std[fd_std < 1e-8] = 1.0   # avoid division by zero on near-constant dims
        torch.save({"mean": torch.tensor(fd_mean), "std": torch.tensor(fd_std)}, norm_path)
        print(f"\n  Feature norm stats saved: {norm_path}")
    else:
        # Load from disk if training set was skipped
        norm = torch.load(norm_path, weights_only=True)
        fd_mean = norm["mean"].numpy()
        fd_std  = norm["std"].numpy()
        print(f"\n  Loaded feature norm stats from {norm_path}")

    # ── Save each split with normalised features ───────────────────────────────
    for split_name, n_samples, seed, filename in splits:
        out_path = os.path.join(DATA_DIR, filename)
        if os.path.exists(out_path) and not args.force and split_name not in raw:
            continue

        if split_name not in raw:
            continue

        r = raw[split_name]
        features_norm = (r["features_raw"] - fd_mean) / fd_std   # z-score
        features_norm = np.clip(features_norm, -5.0, 5.0)         # clip outliers so tanh never saturates

        data = {
            "features":        torch.tensor(features_norm),
            "features_raw":    torch.tensor(r["features_raw"]),
            "targets":         torch.tensor(r["targets"]),
            "trajectories_x":  torch.tensor(r["trajs_x"]),
            "trajectories_y":  torch.tensor(r["trajs_y"]),
            "num_samples":     n_samples,
            "num_fourier":     NUM_FOURIER,
            "num_traj_points": NUM_TRAJ_POINTS,
            "feature_dim":     features_norm.shape[1],
            "target_dim":      6,
            "target_columns":  ["norm_a","norm_b","norm_c","norm_d","norm_p","p_angle"],
            "seed":            seed,
            "split":           split_name,
            "mechanism_type":  "crank_rocker_only",
        }
        torch.save(data, out_path)

        f = data["features"]
        t = data["targets"]
        print(f"\n  Saved: {filename}")
        print(f"    features (normalised) : {tuple(f.shape)}  [{f.min():.2f}, {f.max():.2f}]")
        print(f"    targets               : {tuple(t.shape)}  mean={t.mean():.3f}  std={t.std():.3f}")

    print("\n  All splits complete.")


if __name__ == "__main__":
    main()
