"""
model.py -- Four-bar CRANK-ROCKER linkage MLP predictor.

Architecture:
    Backbone : Linear(30, 256) -> Tanh -> Linear(256,256) -> Tanh -> Linear(256,256) -> Tanh
    Head     : OutputHead (see below)

OutputHead:
    ratio_head : Linear(256, 5) -> 0.05 + sigmoid(x)*0.95
                   => norm_a in (0.05, 1.0)  [crank ALWAYS smallest so norm_a small]
                   => norm_b, norm_c, norm_d, norm_p in (0.05, 1.0)
    angle_head : Linear(256, 1) -> tanh(x) * pi  => p_angle in (-pi, pi)

Changes vs previous version:
    - ReLU -> Tanh backbone (matches reference paper, better for bounded outputs)
    - Hidden dim 128 -> 256 (more capacity for crank-rocker-only problem)
    - Follows paper: tanh activations throughout
"""

import torch
import torch.nn as nn


class OutputHead(nn.Module):
    """
    Decoupled output head:
      - 5 ratio neurons: 0.05 + sigmoid(x)*0.95, bounded to (0.05, 1.0)
        Matches training distribution: min link = 0.5, max possible = 10.0 => min ratio = 0.05
      - 1 tanh-scaled neuron for p_angle in (-pi, pi).
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.ratio_head = nn.Linear(hidden_dim, 5)
        self.angle_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ratios = 0.05 + torch.sigmoid(self.ratio_head(x)) * 0.95  # (0.05, 1.0)
        angle  = torch.tanh(self.angle_head(x)) * torch.pi        # (-pi, pi)
        return torch.cat([ratios, angle], dim=-1)                  # [B, 6]


class LinkagePredictorModel(nn.Module):
    """
    Predicts the four-bar CRANK-ROCKER linkage dimensions from the
    Fourier Descriptors of its coupler point path.

    Input  : [batch, num_fourier_features * 2]   normalised to [-1, 1] in training
    Output : [batch, 6]
               [0] norm_a  -- crank / max_link  in (0.05, 1.0)  (always smallest)
               [1] norm_b  -- coupler / max_link in (0.05, 1.0)
               [2] norm_c  -- rocker / max_link  in (0.05, 1.0)
               [3] norm_d  -- ground / max_link  in (0.05, 1.0)
               [4] norm_p  -- coupler-pt / max_link in (0.05, 1.0)
               [5] p_angle -- coupler-pt angle (rad) in (-pi, pi)
    """

    def __init__(self, num_fourier_features: int = 15, hidden_dim: int = 256):
        super().__init__()
        self.input_dim  = num_fourier_features * 2
        self.output_dim = 6

        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.head = OutputHead(hidden_dim)

    def forward(self, fd: torch.Tensor) -> torch.Tensor:
        """
        fd : [batch_size, num_fourier_features * 2]  (normalised to [-1, 1])
        returns : [batch_size, 6]
        """
        return self.head(self.backbone(fd))


if __name__ == "__main__":
    print("Testing updated model (tanh backbone, 256 hidden, OutputHead)...")
    model = LinkagePredictorModel(num_fourier_features=15, hidden_dim=256)
    dummy_fd = torch.randn(4, 30)
    out = model(dummy_fd)
    ratios = out[:, :5]
    angles = out[:, 5]
    print(f"  Output shape   : {out.shape}")
    print(f"  Ratios range   : [{ratios.min():.3f}, {ratios.max():.3f}]  (should be in (0.05, 1.0))")
    print(f"  p_angle range  : [{angles.min():.3f}, {angles.max():.3f}]  (should be in (-pi, pi))")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params   : {total_params:,}")
    print("  Forward pass OK")
