import torch
import json
import os
from src.train import run_experiment

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_runs = 2
    epochs = 40
    num_samples = 10000
    sparsity = 0.1
    k = 80
    d = 16
    top_k_val = int(sparsity * k)
    
    all_results = []
    
    configs = [
        {"name": "Hard Top-K", "beta": 0.0}, # We'll handle this in the run script if possible
        {"name": "Soft Top-K (Beta 0.1)", "beta": 0.1},
        {"name": "Soft Top-K (Beta 0.5)", "beta": 0.5},
    ]
    
    # Note: Our FeatureDecoder now uses sinkhorn_topk when top_k is set.
    # To simulate Hard Top-K, we can use a very small beta or a different path.
    # For this test, we compare different beta temperatures.
    
    for cfg in configs:
        print(f"\n=== Testing Sinkhorn Temperature: {cfg['name']} ===")
        results = run_experiment(
            k, d, num_samples, sparsity, num_runs, epochs, device, 
            ortho_lambda=0.1, 
            top_k=top_k_val,
            iterative_steps=1,
            beta=cfg['beta'] if cfg['beta'] > 0 else 0.01 # Very small beta for "hard"
        )
        for r in results:
            r["config_name"] = cfg["name"]
        all_results.extend(results)
        
    os.makedirs("results", exist_ok=True)
    with open("results/sinkhorn_test.json", "w") as f:
        json.dump(all_results, f, indent=2)
        
    print("\nSinkhorn tests finished. Results saved to results/sinkhorn_test.json")

if __name__ == "__main__":
    main()
