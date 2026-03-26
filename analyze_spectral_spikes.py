import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from src.train import train_model

def analyze_spectral_spikes(k, d, num_samples=12000, sparsity=0.05, epochs=60, ortho_lambda=0.1, iterative_steps=1, beta=0.1, tau=1.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Spectral Analysis (k={k}, d={d}, alpha={k/d:.2f}, sparsity={sparsity}) ---")
    
    # 1. Train the model
    model, history, F = train_model(
        k, d, num_samples, sparsity, epochs=epochs, 
        device=device, ortho_lambda=ortho_lambda, 
        iterative_steps=iterative_steps, beta=beta, tau=tau
    )
    print(f"Final Task Loss: {history['task_loss'][-1]:.6f}")
    
    # 2. Extract normalized W and F
    W = model.get_normalized_W().detach().cpu()
    F = F.detach().cpu()

    # Calculate Direct Alignment: Signal = mean(diag(W @ F.T)), Interference = mean(off-diag(W @ F.T))
    # This is the most direct proof of feature recovery.
    alignment_matrix = torch.matmul(W, F.t())
    avg_signal = torch.diagonal(alignment_matrix).mean().item()
    
    mask = torch.eye(k).bool()
    avg_interference = alignment_matrix[~mask].abs().mean().item()
    
    print(f"Average Signal Alignment <w_i, f_i>: {avg_signal:.4f}")
    print(f"Average Interference <w_i, f_j>: {avg_interference:.4f}")
    print(f"Signal-to-Interference Ratio (Direct): {avg_signal / (avg_interference + 1e-9):.2f}")
    
    print(f"Mean Norm of W rows (normalized): {torch.norm(W, dim=1).mean():.4f}")
    print(f"Mean Norm of F rows: {torch.norm(F, dim=1).mean():.4f}")

    # 3. Calculate the Sample Covariance Matrix: Sigma = (1/k) * W * F.T * F * W.T
    X = torch.matmul(W, F.t()) # (k, k)
    Sigma = torch.matmul(X, X.t()) / k
    
    # 4. Compute Eigenvalues
    eigenvalues = torch.linalg.eigvalsh(Sigma)
    eigenvalues = eigenvalues.numpy()
    
    # 5. Theoretical Marchenko-Pastur Edge
    # alpha = k/d
    alpha = k / d
    mp_edge = (1 + np.sqrt(alpha))**2
    
    # 6. Identify Spikes
    spikes = eigenvalues[eigenvalues > mp_edge]
    num_spikes = len(spikes)
    
    print(f"Marchenko-Pastur Edge (Theoretical): {mp_edge:.4f}")
    print(f"Max Eigenvalue: {np.max(eigenvalues):.4f}")
    print(f"Number of Signal Spikes (> Edge): {num_spikes}")
    print(f"Spike Ratio (Spikes / k): {num_spikes / k:.2%}")
    
    # 7. Visualization
    plt.figure(figsize=(10, 6))
    plt.hist(eigenvalues, bins=50, density=True, alpha=0.7, label='Empirical ESD')
    plt.axvline(mp_edge, color='r', linestyle='--', label=f'MP Edge ({mp_edge:.2f})')
    plt.title(f'Spectral Distribution (k={k}, d={d}, alpha={alpha:.2f})')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Density')
    plt.legend()
    
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/spectral_spikes.png")
    print("Spectral plot saved to plots/spectral_spikes.png")
    
    return num_spikes, mp_edge

if __name__ == "__main__":
    # Test with higher dimension and lower sparsity to force spikes
    # k=120 features in d=40 dimensions (alpha=3)
    analyze_spectral_spikes(k=120, d=40, sparsity=0.05)
