import json
import pandas as pd
import os

def main():
    with open("results/causal_test.json", "r") as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    
    # Average over runs
    df_avg = df.groupby(['k', 'd', 'ortho_lambda']).mean(numeric_only=True).reset_index()
    
    print("--- Causal Test Results ---")
    cols = ['k', 'd', 'ortho_lambda', 'final_task_loss', 'final_test_loss', 'final_signal', 'final_interference', 'final_I_S_ratio']
    print(df_avg[cols].to_string(index=False))
    
    # Check H2
    # For high density k=100, d=10
    high_density = df_avg[df_avg['k'] == 100]
    baseline_loss = high_density[high_density['ortho_lambda'] == 0.0]['final_test_loss'].values[0]
    reg_loss = high_density[high_density['ortho_lambda'] > 0.0]['final_test_loss'].min()
    
    if reg_loss < baseline_loss:
        print(f"\nH2 SUPPORTED: Regularization (I/S reduction) improved test loss from {baseline_loss:.6f} to {reg_loss:.6f}")
    else:
        print(f"\nH2 NOT SUPPORTED: Regularization did not improve test loss (Baseline: {baseline_loss:.6f}, Reg: {reg_loss:.6f})")

if __name__ == "__main__":
    main()
