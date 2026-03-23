import torch
import json
import os
from src.train import run_experiment

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_runs = 3
    epochs = 50
    num_samples = 10000
    sparsity = 0.1
    k = 50
    d = 10
    top_k_val = int(sparsity * k)
    
    all_results = []
    
    configs = [
        {"name": "Baseline", "lmb": 0.0, "top_k": None},
        {"name": "Regularization", "lmb": 0.1, "top_k": None},
        {"name": "Sparsity", "lmb": 0.0, "top_k": top_k_val},
        {"name": "Both", "lmb": 0.1, "top_k": top_k_val},
    ]
    
    for cfg in configs:
        print(f"--- Comparing: {cfg['name']} ---")
        results = run_experiment(k, d, num_samples, sparsity, num_runs, epochs, device, 
                                 ortho_lambda=cfg['lmb'], top_k=cfg['top_k'])
        # Add name to each result
        for r in results:
            r["config_name"] = cfg["name"]
        all_results.extend(results)
        
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/comparison_test.json", "w") as f:
        json.dump(all_results, f, indent=2)
        
    print("Comparison test finished. Results saved to results/comparison_test.json")

if __name__ == "__main__":
    main()
