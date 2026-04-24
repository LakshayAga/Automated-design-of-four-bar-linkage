"""
Full training script with validation monitoring, early stopping, and wrapped angle loss.

Usage (from project root):
    python src/train_full.py
    python src/train_full.py --epochs 1000 --patience 40 --lr 1e-3 --batch 128

Requires:
    data/train_dataset.pt
    data/val_dataset.pt

Outputs:
    models/best_model.pth
    models/best_model_meta.pt
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import LinkagePredictorModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")


def wrapped_angle_mse(pred_angle: torch.Tensor, true_angle: torch.Tensor) -> torch.Tensor:
    """
    MSE loss for an angle in (-pi, pi) that accounts for wrapping.
    The shortest angular distance between two angles is used.
    d = atan2(sin(pred - true), cos(pred - true)) is in (-pi, pi).
    """
    diff = pred_angle - true_angle
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    return (diff ** 2).mean()


def linkage_loss(pred: torch.Tensor, target: torch.Tensor,
                 angle_weight: float = 0.5) -> torch.Tensor:
    """
    Combined loss:
      - MSE on the 5 normalised link-length ratios (indices 0-4)
      - Wrapped MSE on the coupler-point angle (index 5)
    angle_weight controls the relative contribution of the angle term.
    """
    ratio_loss = nn.functional.mse_loss(pred[:, :5], target[:, :5])
    angle_loss = wrapped_angle_mse(pred[:, 5], target[:, 5])
    return ratio_loss + angle_weight * angle_loss


def load_split(filename: str):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Run  python src/generate_full_dataset.py  first."
        )
    d = torch.load(path, weights_only=True)
    return d["features"], d["targets"]


def main():
    parser = argparse.ArgumentParser(description="Train the four-bar crank-rocker predictor.")
    parser.add_argument("--epochs",       type=int,   default=1000)
    parser.add_argument("--batch",        type=int,   default=128)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--patience",     type=int,   default=40)
    parser.add_argument("--hidden",       type=int,   default=256)
    parser.add_argument("--angle-weight", type=float, default=0.5,
                        help="Weight for the wrapped angle loss term (default 0.5).")
    args = parser.parse_args()

    torch.manual_seed(42)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading datasets...")
    train_X, train_y = load_split("train_dataset.pt")
    val_X,   val_y   = load_split("val_dataset.pt")
    print(f"  Train : {tuple(train_X.shape)}  Val : {tuple(val_X.shape)}")

    train_loader = DataLoader(
        TensorDataset(train_X, train_y), batch_size=args.batch, shuffle=True,  drop_last=False
    )
    val_loader = DataLoader(
        TensorDataset(val_X, val_y),     batch_size=args.batch, shuffle=False, drop_last=False
    )

    # ── Model, optimiser, scheduler ───────────────────────────────────────────
    model     = LinkagePredictorModel(num_fourier_features=15, hidden_dim=args.hidden)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5   # cut LR quickly to avoid overshoot spikes
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params:,}  |  angle_weight={args.angle_weight}")

    best_val_loss  = float("inf")
    best_epoch     = 0
    patience_count = 0
    loss_history   = {"train": [], "val": []}
    model_path     = os.path.join(MODEL_DIR, "best_model.pth")
    meta_path      = os.path.join(MODEL_DIR, "best_model_meta.pt")

    header = f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}  {'Best Val':>12}  {'LR':>10}"
    print(f"\nTraining (max={args.epochs} epochs, patience={args.patience})\n{header}")
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = linkage_loss(model(xb), yb, args.angle_weight)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_X)

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += linkage_loss(model(xb), yb, args.angle_weight).item() * xb.size(0)
        val_loss /= len(val_X)

        scheduler.step(val_loss)
        loss_history["train"].append(train_loss)
        loss_history["val"].append(val_loss)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss  = val_loss
            best_epoch     = epoch
            patience_count = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_count += 1

        lr_now = optimizer.param_groups[0]["lr"]
        star   = " <-- best" if improved else ""
        if epoch % 20 == 0 or epoch == 1 or improved:
            print(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  "
                  f"{best_val_loss:>12.6f}  {lr_now:>10.2e}{star}")

        if patience_count >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs).")
            break

    # ── Save metadata ──────────────────────────────────────────────────────────
    torch.save({
        "loss_history":  loss_history,
        "best_val_loss": best_val_loss,
        "best_epoch":    best_epoch,
        "epochs_run":    epoch,
        "batch_size":    args.batch,
        "learning_rate": args.lr,
        "patience":      args.patience,
        "hidden_dim":    args.hidden,
        "angle_weight":  args.angle_weight,
        "architecture":  str(model),
    }, meta_path)

    print(f"\nTraining complete.")
    print(f"  Best epoch   : {best_epoch}")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Checkpoint   : {model_path}")


if __name__ == "__main__":
    main()
