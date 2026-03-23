import torch
import torch.nn as nn
from src.core import FeatureDecoder
from src.dataset import get_dataloader

def measure_stability(model, k, d, num_samples=1000, sparsity=0.1, noise_std=0.01, device='cpu'):
    model.eval()
    dataloader, _ = get_dataloader(k, d, num_samples, sparsity, batch_size=64, seed=999)
    
    total_delta_y = 0
    total_samples = 0
    
    with torch.no_grad():
        for h, a in dataloader:
            h = h.to(device)
            # Original output
            y = model(h)
            
            # Noisy input
            noise = torch.randn_like(h) * noise_std
            y_noisy = model(h + noise)
            
            # Δy = ||y(x + ε) − y(x)||
            delta_y = torch.norm(y_noisy - y, dim=1)
            total_delta_y += torch.sum(delta_y).item()
            total_samples += h.size(0)
            
    return total_delta_y / total_samples

if __name__ == "__main__":
    # This could be integrated into the analysis scripts
    pass
