import json
import pandas as pd
import os

def main():
    with open("results/comparison_test.json", "r") as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    
    # Average over runs
    df_avg = df.groupby(['config_name']).mean(numeric_only=True).reset_index()
    
    print("--- Model Comparison Results (k=50, d=10) ---")
    cols = ['config_name', 'final_test_loss', 'stability', 'final_I_S_ratio', 'final_signal', 'final_interference']
    # Reorder to match the comparison table in Phase 5
    df_avg = df_avg.set_index('config_name').loc[['Baseline', 'Regularization', 'Sparsity', 'Both']].reset_index()
    print(df_avg[cols].to_string(index=False))
    
    # Conclusion
    best_config = df_avg.loc[df_avg['final_test_loss'].idxmin()]
    print(f"\nBEST MODEL: {best_config['config_name']} with test loss {best_config['final_test_loss']:.6f}")
    
    baseline_stability = df_avg[df_avg['config_name'] == 'Baseline']['stability'].values[0]
    best_stability = df_avg.loc[df_avg['stability'].idxmin()]
    print(f"STABILITY IMPROVEMENT: Baseline {baseline_stability:.6f} vs Best ({best_stability['config_name']}) {best_stability['stability']:.6f}")

if __name__ == "__main__":
    main()
