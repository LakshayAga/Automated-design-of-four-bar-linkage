"""
Generate the MSE training plot from ACTUAL stored training metadata.

Reads the loss history from models/preliminary_training_meta.pt
and plots the real training curve.

Run from project root:
    python scripts/generate_plot.py
"""

import os
import torch
import matplotlib.pyplot as plt

# Resolve paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
META_PATH = os.path.join(PROJECT_ROOT, "models", "preliminary_training_meta.pt")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")

# Load actual training metadata
meta = torch.load(META_PATH, weights_only=True)
loss_history = meta['loss_history']
num_epochs = meta['epochs']
dataset_size = meta['dataset_size']
lr = meta['learning_rate']

epochs = range(1, num_epochs + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, loss_history, marker='o', markersize=4, linestyle='-',
         color='#007acc', label='Training MSE')

plt.title('Training Loss (MSE) vs Epochs - Preliminary Model')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (MSE)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Annotate final loss
plt.annotate(f'Final: {loss_history[-1]:.4f}',
             xy=(num_epochs, loss_history[-1]),
             xytext=(-80, 20), textcoords='offset points',
             fontsize=9, color='#007acc',
             arrowprops=dict(arrowstyle='->', color='#007acc'))

# Subtitle with hyperparameters
plt.figtext(0.5, 0.01,
            f'Dataset: {dataset_size} samples | LR: {lr} | Epochs: {num_epochs}',
            ha='center', fontsize=8, color='gray')

plt.tight_layout(rect=[0, 0.03, 1, 1])

os.makedirs(ASSETS_DIR, exist_ok=True)
out_path = os.path.join(ASSETS_DIR, "mse_training_plot.png")
plt.savefig(out_path, dpi=300)
print(f"Plot generated from actual training data at {out_path}")
