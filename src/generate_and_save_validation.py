"""
Generate and persist the initial validation dataset and trained model.

Purpose:
    This script is used for SELF-VALIDATION of the data generation pipeline
    and the model architecture. It generates a small batch of training data,
    saves it to disk, trains the initial MLP model, and saves the resulting
    checkpoint. These artifacts can be used to verify that:
        1. The data generation (forward kinematics + Fourier descriptors) works correctly.
        2. The model can learn from the generated data (loss converges).
        3. The saved model can be loaded and used for inference.

Outputs:
    data/validation_dataset.pt       — Features (Fourier descriptors), targets (linkage params),
                                        AND raw (x, y) coupler-point trajectories for every sample.
    models/validation_model.pth      — The trained model state dict
    models/validation_training_meta.pt — Training metadata (loss history, hyperparams)
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_generation import (
    is_grashof,
    forward_kinematics_trajectory,
    compute_fourier_descriptors,
)
from model import LinkagePredictorModel

# ─── Configuration ───────────────────────────────────────────────────────────
DATASET_SIZE = 500       # Small batch for initial validation
NUM_EPOCHS = 50          # Enough to confirm convergence
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_FOURIER = 15         # Matching the existing pipeline
NUM_TRAJ_POINTS = 128    # Points per coupler-point trajectory
RANDOM_SEED = 42

# Output paths (relative to project root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


def main():
    # Set seeds for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Create output directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Step 1: Generate and save the validation dataset ─────────────────
    print("=" * 60)
    print("STEP 1: Generating validation dataset")
    print("=" * 60)
    print(f"  Dataset size      : {DATASET_SIZE}")
    print(f"  Fourier terms     : {NUM_FOURIER}")
    print(f"  Trajectory points : {NUM_TRAJ_POINTS}")
    print()

    # --- Custom generation loop (captures raw trajectories) ---------------
    from tqdm import tqdm

    fd_list = []          # Fourier descriptor vectors  (30-dim each)
    params_list = []      # Normalized linkage params    (6-dim each)
    trajectories_x = []   # Raw coupler-point X coords   (128-dim each)
    trajectories_y = []   # Raw coupler-point Y coords   (128-dim each)

    pbar = tqdm(total=DATASET_SIZE)
    while len(fd_list) < DATASET_SIZE:
        a, b, c, d = np.random.uniform(0.5, 10.0, size=4)
        p_dist = np.random.uniform(0.5, 10.0)
        p_angle = np.random.uniform(-np.pi, np.pi)

        if not is_grashof(a, b, c, d):
            continue

        res = forward_kinematics_trajectory(
            a, b, c, d, p_dist, p_angle, num_points=NUM_TRAJ_POINTS
        )
        if res is None:
            continue

        Px, Py = res
        if np.any(np.isnan(Px)) or np.any(np.isnan(Py)):
            continue

        fd = compute_fourier_descriptors(Px, Py, num_descriptors=NUM_FOURIER)

        # Normalize dimensions (same convention as data_generation.py)
        max_len = max(a, b, c, d)
        target = np.array([
            a / max_len, b / max_len, c / max_len,
            d / max_len, p_dist / max_len, p_angle,
        ])

        fd_list.append(fd)
        params_list.append(target)
        trajectories_x.append(Px)
        trajectories_y.append(Py)
        pbar.update(1)

    pbar.close()

    features = torch.tensor(np.array(fd_list), dtype=torch.float32)
    targets = torch.tensor(np.array(params_list), dtype=torch.float32)
    traj_x = torch.tensor(np.array(trajectories_x), dtype=torch.float32)
    traj_y = torch.tensor(np.array(trajectories_y), dtype=torch.float32)

    dataset_path = os.path.join(DATA_DIR, "validation_dataset.pt")
    torch.save({
        'features': features,           # [N, 30]  Fourier descriptors
        'targets': targets,             # [N, 6]   normalized linkage params
        'trajectories_x': traj_x,       # [N, 128] raw coupler X coordinates
        'trajectories_y': traj_y,       # [N, 128] raw coupler Y coordinates
        'num_samples': DATASET_SIZE,
        'num_fourier': NUM_FOURIER,
        'num_traj_points': NUM_TRAJ_POINTS,
        'feature_dim': features.shape[1],
        'target_dim': targets.shape[1],
        'target_columns': ['norm_a', 'norm_b', 'norm_c', 'norm_d', 'norm_p', 'p_angle'],
        'seed': RANDOM_SEED,
    }, dataset_path)

    print(f"\n  Dataset saved to: {dataset_path}")
    print(f"  Features shape      : {features.shape}")
    print(f"  Targets shape       : {targets.shape}")
    print(f"  Trajectories shape  : X={traj_x.shape}, Y={traj_y.shape}")
    print(f"  Feature stats       : mean={features.mean():.4f}, std={features.std():.4f}")
    print(f"  Target stats        : mean={targets.mean():.4f}, std={targets.std():.4f}")

    # ── Step 2: Train initial model and save ─────────────────────────────
    print()
    print("=" * 60)
    print("STEP 2: Training initial validation model")
    print("=" * 60)
    print(f"  Epochs        : {NUM_EPOCHS}")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Learning rate : {LEARNING_RATE}")
    print()

    tensor_dataset = TensorDataset(features, targets)
    dataloader = DataLoader(tensor_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = LinkagePredictorModel(num_fourier_features=NUM_FOURIER)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loss_history = []

    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for batch_fd, batch_target in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_fd)
            loss = criterion(predictions, batch_target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_fd.size(0)

        epoch_loss /= DATASET_SIZE
        loss_history.append(epoch_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{NUM_EPOCHS} | MSE Loss: {epoch_loss:.6f}")

    print(f"\n  Final loss: {loss_history[-1]:.6f}")

    # Save model weights
    model_path = os.path.join(MODEL_DIR, "validation_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"  Model saved to: {model_path}")

    # Save training metadata
    meta_path = os.path.join(MODEL_DIR, "validation_training_meta.pt")
    torch.save({
        'loss_history': loss_history,
        'final_loss': loss_history[-1],
        'epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'dataset_size': DATASET_SIZE,
        'architecture': str(model),
        'seed': RANDOM_SEED,
    }, meta_path)
    print(f"  Metadata saved to: {meta_path}")

    # ── Step 3: Quick self-validation ────────────────────────────────────
    print()
    print("=" * 60)
    print("STEP 3: Self-validation (load and verify)")
    print("=" * 60)

    # Reload the model from disk
    loaded_model = LinkagePredictorModel(num_fourier_features=NUM_FOURIER)
    loaded_model.load_state_dict(torch.load(model_path, weights_only=True))
    loaded_model.eval()

    # Reload the dataset from disk
    loaded_data = torch.load(dataset_path, weights_only=True)
    loaded_features = loaded_data['features']
    loaded_targets = loaded_data['targets']

    # Run a quick inference check on the first 5 samples
    with torch.no_grad():
        sample_features = loaded_features[:5]
        sample_targets = loaded_targets[:5]
        sample_preds = loaded_model(sample_features)

        print(f"\n  Sample predictions vs ground truth (first 5):")
        print(f"  {'':>4}  {'Predicted':>50}  {'Actual':>50}")
        for i in range(5):
            pred_str = ", ".join(f"{v:.3f}" for v in sample_preds[i].numpy())
            actual_str = ", ".join(f"{v:.3f}" for v in sample_targets[i].numpy())
            print(f"  [{i}]  [{pred_str}]  [{actual_str}]")

        # Compute validation MSE on all data
        all_preds = loaded_model(loaded_features)
        val_mse = nn.MSELoss()(all_preds, loaded_targets).item()
        print(f"\n  Reloaded-model validation MSE: {val_mse:.6f}")

    print()
    print("=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print(f"  Stored artifacts:")
    print(f"    Data  : {dataset_path}")
    print(f"      -> contains: features (FDs), targets, trajectories_x, trajectories_y")
    print(f"    Model : {model_path}")
    print(f"    Meta  : {meta_path}")
    print()


if __name__ == "__main__":
    main()
