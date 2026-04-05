import torch
import torch.nn as nn

class LinkagePredictorModel(nn.Module):
    def __init__(self, num_fourier_features: int = 15, hidden_dim: int = 128):
        """
        Predicts the four-bar linkage dimensions given the Fourier Descriptors of its path.
        num_fourier_features: Number of complex FD components extracted from path.
        The input tensor is expected to be flattened [real, imag] arrays, 
        so the actual input dimension is num_fourier_features * 2.
        """
        super().__init__()
        
        self.input_dim = num_fourier_features * 2
        # Predicting: norm_a, norm_b, norm_c, norm_d, norm_p, p_angle
        self.output_dim = 6 
        
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim)
        )
        
    def forward(self, fd: torch.Tensor) -> torch.Tensor:
        """
        fd: [batch_size, num_fourier_features * 2]
        returns: [batch_size, 6] (predicted normalized parameters)
        """
        return self.net(fd)

if __name__ == "__main__":
    print("Testing ML model interface...")
    model = LinkagePredictorModel(num_fourier_features=15)
    # Dummy batch of 4 examples
    dummy_fd = torch.randn(4, 15 * 2)
    out = model(dummy_fd)
    print(f"Input shape: {dummy_fd.shape}")
    print(f"Output shape: {out.shape}")
    print("Forward pass successful.")
