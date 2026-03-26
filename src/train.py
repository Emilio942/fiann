import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
import os
from src.core import FeatureDecoder, calculate_interference_metrics
from src.dataset import get_dataloader

def train_model(num_features, feature_dim, num_samples, sparsity, epochs=20, lr=0.01, seed=42, device='cpu', ortho_lambda=0.0, top_k=None, threshold=None, iterative_steps=0, beta=0.1, tau=1.0):
    torch.manual_seed(seed)
    
    # Get data and feature matrix
    dataloader, F = get_dataloader(num_features, feature_dim, num_samples, sparsity, batch_size=64, seed=seed)
    # Create a separate test set
    test_dataloader, _ = get_dataloader(num_features, feature_dim, num_samples // 5, sparsity, batch_size=64, seed=seed + 1000)
    
    F = F.to(device)
    
    # Initialize model with advanced parameters
    model = FeatureDecoder(feature_dim, num_features, top_k=top_k, threshold=threshold, iterative_steps=iterative_steps, beta=beta, tau=tau).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    history = {
        "loss": [],
        "task_loss": [],
        "ortho_loss": [],
        "triple_loss": [],
        "test_loss": [],
        "signal": [],
        "interference": [],
        "I_S_ratio": [],
        "interference_entropy": [],
        "condition_number": [],
        "ortho_error": []
    }
    
    for epoch in range(epochs):
        model.train()
        epoch_task_loss = 0
        epoch_ortho_loss = 0
        epoch_triple_loss = 0
        for h, a in dataloader:
            h, a = h.to(device), a.to(device)
            
            optimizer.zero_grad()
            y = model(h)
            task_loss = criterion(y, a)
            
            # 1. Squared Orthogonality Regularization (L2-style)
            # Inquiry 4: Absolute value was acting as aggressive pruning.
            # Using normalized weights for the penalty.
            W_norm = model.get_normalized_W()
            dot_products_sq = torch.square(torch.matmul(W_norm, F.t()))
            mask = torch.eye(num_features, device=device).bool()
            ortho_reg = torch.mean(dot_products_sq[~mask])
            
            # 2. Triple Interaction Regularization
            with torch.no_grad():
                idx_m = torch.randint(0, num_features, (num_features,), device=device)
                idx_l = torch.randint(0, num_features, (num_features,), device=device)
                f_mixed = F[idx_m] * F[idx_l]
            
            # Use normalized weights for triple reg
            triple_prod = torch.abs(torch.sum(W_norm * f_mixed, dim=1))
            triple_reg = torch.mean(triple_prod)
            
            # Total Loss
            loss = task_loss + ortho_lambda * ortho_reg + (ortho_lambda * 0.1) * triple_reg
            loss.backward()
            optimizer.step()
            
            epoch_task_loss += task_loss.item()
            epoch_ortho_loss += ortho_reg.item()
            epoch_triple_loss += triple_reg.item()
            
        avg_task_loss = epoch_task_loss / len(dataloader)
        avg_ortho_loss = epoch_ortho_loss / len(dataloader)
        avg_triple_loss = epoch_triple_loss / len(dataloader)
        avg_total_loss = avg_task_loss + ortho_lambda * avg_ortho_loss
        
        # Test evaluation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for h, a in test_dataloader:
                h, a = h.to(device), a.to(device)
                y = model(h)
                test_loss += criterion(y, a).item()
        avg_test_loss = test_loss / len(test_dataloader)
        
        # Calculate metrics for this epoch (using normalized weights)
        metrics = calculate_interference_metrics(model.get_normalized_W().data, F)
        
        history["loss"].append(avg_total_loss)
        history["task_loss"].append(avg_task_loss)
        history["ortho_loss"].append(avg_ortho_loss)
        history["triple_loss"].append(avg_triple_loss)
        history["test_loss"].append(avg_test_loss)
        history["signal"].append(metrics["signal"])
        history["interference"].append(metrics["interference"])
        history["I_S_ratio"].append(metrics["I_S_ratio"])
        history["interference_entropy"].append(metrics["interference_entropy"])
        history["condition_number"].append(metrics["condition_number"])
        history["ortho_error"].append(metrics["ortho_error"])
        
    return model, history, F

from src.stability import measure_stability

def run_experiment(k, d, num_samples=10000, sparsity=0.1, num_runs=3, epochs=20, device='cpu', ortho_lambda=0.0, top_k=None, threshold=None, iterative_steps=0, beta=0.1, tau=1.0):
    results = []
    for run in range(num_runs):
        seed = 42 + run
        print(f"Starting Run {run+1}/{num_runs} with seed {seed} (k={k}, d={d}, lambda={ortho_lambda}, top_k={top_k}, iterative={iterative_steps}, beta={beta}, tau={tau})")
        model, history, F = train_model(k, d, num_samples, sparsity, epochs, seed=seed, device=device, ortho_lambda=ortho_lambda, top_k=top_k, threshold=threshold, iterative_steps=iterative_steps, beta=beta, tau=tau)
        
        # Stability: Δy = ||y(x + ε) − y(x)||
        stability_noise = 0.01
        stability = measure_stability(model, k, d, num_samples // 10, sparsity, noise_std=stability_noise, device=device)
        
        # Save final metrics
        final_metrics = {
            "run": run,
            "seed": seed,
            "k": k,
            "d": d,
            "ortho_lambda": ortho_lambda,
            "top_k": top_k,
            "threshold": threshold,
            "iterative_steps": iterative_steps,
            "beta": beta,
            "tau": tau,
            "final_loss": history["loss"][-1],
            "final_task_loss": history["task_loss"][-1],
            "final_ortho_loss": history["ortho_loss"][-1],
            "final_triple_loss": history["triple_loss"][-1],
            "final_test_loss": history["test_loss"][-1],
            "final_signal": history["signal"][-1],
            "final_interference": history["interference"][-1],
            "final_I_S_ratio": history["I_S_ratio"][-1],
            "final_entropy": history["interference_entropy"][-1],
            "final_cond": history["condition_number"][-1],
            "final_ortho": history["ortho_error"][-1],
            "stability": stability
        }
        results.append(final_metrics)
        
    return results

if __name__ == "__main__":
    # Test run
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = run_experiment(k=20, d=10, num_runs=1, device=device)
    print(json.dumps(results, indent=2))
