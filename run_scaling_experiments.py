import torch
import json
import os
import pandas as pd
from src.train import run_experiment

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_runs = 3
    epochs = 30 # A bit more for better convergence
    num_samples = 10000
    sparsity = 0.1
    
    all_results = []
    
    # Experiment A: k ↑, d constant (10)
    print("--- Experiment A: k ↑, d=10 ---")
    d_const = 10
    k_values = [10, 20, 50, 100, 200]
    for k in k_values:
        results = run_experiment(k, d_const, num_samples, sparsity, num_runs, epochs, device)
        all_results.extend(results)
        
    # Experiment B: d ↑, k constant (50)
    print("--- Experiment B: d ↑, k=50 ---")
    k_const = 50
    d_values = [5, 10, 25, 50, 100]
    for d in d_values:
        # Avoid duplicate (50, 10) if already done in Exp A
        if d == 10: continue 
        results = run_experiment(k_const, d, num_samples, sparsity, num_runs, epochs, device)
        all_results.extend(results)
        
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/scaling_experiments.json", "w") as f:
        json.dump(all_results, f, indent=2)
        
    print("Experiments finished. Results saved to results/scaling_experiments.json")

if __name__ == "__main__":
    main()
