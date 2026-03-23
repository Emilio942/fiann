import torch
import json
import os
from src.train import run_experiment

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_runs = 3
    epochs = 50 # More epochs to see regularization effect
    num_samples = 10000
    sparsity = 0.1
    
    all_results = []
    
    # High density case
    k_high = 100
    d = 10
    print(f"--- Causal Test: High Density (k={k_high}, d={d}) ---")
    for lmb in [0.0, 0.1, 0.5]:
        results = run_experiment(k_high, d, num_samples, sparsity, num_runs, epochs, device, ortho_lambda=lmb)
        all_results.extend(results)
        
    # Low density case
    k_low = 20
    print(f"--- Causal Test: Low Density (k={k_low}, d={d}) ---")
    for lmb in [0.0, 0.1]:
        results = run_experiment(k_low, d, num_samples, sparsity, num_runs, epochs, device, ortho_lambda=lmb)
        all_results.extend(results)
        
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/causal_test.json", "w") as f:
        json.dump(all_results, f, indent=2)
        
    print("Causal test finished. Results saved to results/causal_test.json")

if __name__ == "__main__":
    main()
