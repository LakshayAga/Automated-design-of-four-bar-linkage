"""
View the stored validation data, plot a sample linkage trajectory,
and plot the training loss curve.

Run from the project root:
    python data/view_data.py
"""

import os
import torch
import matplotlib.pyplot as plt

# Resolve paths relative to project root (one level up from this file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "validation_dataset.pt")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "preliminary_model.pth")
META_PATH = os.path.join(PROJECT_ROOT, "models", "preliminary_training_meta.pt")

# ── 1. Load and inspect the dataset ─────────────────────────────────────────
print("=" * 60)
print("DATASET CONTENTS")
print("=" * 60)
data = torch.load(DATASET_PATH, weights_only=True)
print(f"Keys            : {list(data.keys())}")
print(f"Features shape  : {data['features'].shape}")
print(f"Targets shape   : {data['targets'].shape}")
print(f"Traj X shape    : {data['trajectories_x'].shape}")
print(f"Traj Y shape    : {data['trajectories_y'].shape}")
print(f"Num samples     : {data['num_samples']}")
print(f"Num traj points : {data['num_traj_points']}")
print(f"Target columns  : {data['target_columns']}")
print()

# ── 2. Load and inspect model weights ───────────────────────────────────────
print("=" * 60)
print("MODEL WEIGHTS")
print("=" * 60)
weights = torch.load(MODEL_PATH, weights_only=True)
print(f"Layer keys: {list(weights.keys())}")
for key, tensor in weights.items():
    print(f"  {key:25s} -> shape {list(tensor.shape)}")
print()

# ── 3. Load training metadata ───────────────────────────────────────────────
print("=" * 60)
print("TRAINING METADATA")
print("=" * 60)
meta = torch.load(META_PATH, weights_only=True)
for key, val in meta.items():
    if key == 'loss_history':
        print(f"  {key:20s} -> [{val[0]:.4f}, ..., {val[-1]:.4f}] ({len(val)} epochs)")
    else:
        print(f"  {key:20s} -> {val}")
print()

# ── 4. Plot a sample linkage coupler trajectory ─────────────────────────────
SAMPLE_IDX = 0  # Change this to view a different sample

traj_x = data['trajectories_x'][SAMPLE_IDX].numpy()
traj_y = data['trajectories_y'][SAMPLE_IDX].numpy()
params = data['targets'][SAMPLE_IDX].numpy()
col_names = data['target_columns']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Left: coupler trajectory ---
ax1 = axes[0]
ax1.plot(traj_x, traj_y, color='#2563eb', linewidth=1.5, label='Coupler path')
ax1.plot(traj_x[0], traj_y[0], 'o', color='#16a34a', markersize=8, label='Start', zorder=5)
ax1.scatter(traj_x, traj_y, c=range(len(traj_x)), cmap='viridis', s=8, zorder=4, alpha=0.6)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title(f'Coupler-Point Trajectory (Sample #{SAMPLE_IDX})')
ax1.set_aspect('equal')
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.legend()

# Annotate with linkage params
param_text = "\n".join(f"{col_names[i]}: {params[i]:.3f}" for i in range(len(params)))
ax1.text(0.02, 0.98, param_text, transform=ax1.transAxes,
         fontsize=8, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))

# --- Right: training loss curve ---
ax2 = axes[1]
loss = meta['loss_history']
epochs = range(1, len(loss) + 1)
ax2.plot(epochs, loss, color='#dc2626', linewidth=1.8, marker='o', markersize=3)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MSE Loss')
ax2.set_title('Training Loss Curve')
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.annotate(f'Final: {loss[-1]:.4f}',
             xy=(len(loss), loss[-1]),
             xytext=(-60, 20), textcoords='offset points',
             fontsize=9, color='#dc2626',
             arrowprops=dict(arrowstyle='->', color='#dc2626'))

plt.tight_layout()

# Save and show
out_path = os.path.join(PROJECT_ROOT, "assets", "validation_overview.png")
plt.savefig(out_path, dpi=150)
print(f"Plot saved to: {out_path}")
plt.show()
