import torch
import json
import os
from src.train import run_experiment

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_runs = 2 # Reduced for speed
    epochs = 30
    num_samples = 8000
    sparsity = 0.1
    k = 60 # Overcomplete regime
    d = 12
    top_k_val = int(sparsity * k)
    
    all_results = []
    
    configs = [
        {"name": "Baseline (Linear)", "lmb": 0.0, "top_k": None, "iterative": 0, "thr": None},
        {"name": "Regularized (Triple)", "lmb": 0.1, "top_k": None, "iterative": 0, "thr": None},
        {"name": "Advanced (Iterative + Triple)", "lmb": 0.1, "top_k": top_k_val, "iterative": 3, "thr": 0.2},
    ]
    
    for cfg in configs:
        print(f"\n=== Testing Configuration: {cfg['name']} ===")
        results = run_experiment(
            k, d, num_samples, sparsity, num_runs, epochs, device, 
            ortho_lambda=cfg['lmb'], 
            top_k=cfg['top_k'],
            threshold=cfg['thr'],
            iterative_steps=cfg['iterative']
        )
        for r in results:
            r["config_name"] = cfg["name"]
        all_results.extend(results)
        
    os.makedirs("results", exist_ok=True)
    with open("results/advanced_math_test.json", "w") as f:
        json.dump(all_results, f, indent=2)
        
    print("\nAdvanced mathematical tests finished. Results saved to results/advanced_math_test.json")

if __name__ == "__main__":
    main()
