import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# Import from our modules
from data_generation import generate_dataset
from model import LinkagePredictorModel

def train_model(epochs=100, batch_size=32, dataset_size=2000):
    print(f"Generating synthetic dataset of size {dataset_size}...")
    ds = generate_dataset(num_samples=dataset_size)
    
    # Extract features (fd) and targets (params)
    # fd shape is (15 * 2) = 30
    # params shape is (6)
    features = torch.tensor([item['fd'] for item in ds], dtype=torch.float32)
    targets = torch.tensor([item['params'] for item in ds], dtype=torch.float32)
    
    dataset = TensorDataset(features, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = LinkagePredictorModel(num_fourier_features=15)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print("Starting training...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_fd, batch_target in dataloader:
            optimizer.zero_grad()
            
            predictions = model(batch_fd)
            loss = criterion(predictions, batch_target)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_fd.size(0)
            
        epoch_loss /= dataset_size
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | MSE Loss: {epoch_loss:.6f}")
            
    print("Training complete.")
    # torch.save(model.state_dict(), "models/validation_model.pth")
    return model

if __name__ == "__main__":
    train_model(epochs=50, dataset_size=500)
