import json
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    with open("results/scaling_experiments.json", "r") as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    df['k/d'] = df['k'] / df['d']
    
    # Average over runs
    df_avg = df.groupby(['k', 'd', 'k/d']).mean(numeric_only=True).reset_index()
    
    os.makedirs("plots", exist_ok=True)
    
    # 1. Final Interference vs k/d
    plt.figure(figsize=(10, 6))
    plt.scatter(df_avg['k/d'], df_avg['final_interference'], c='blue', label='Avg Interference')
    plt.xlabel('Feature Density (k/d)')
    plt.ylabel('Final Interference (I)')
    plt.title('Interference vs Feature Density')
    plt.grid(True)
    plt.savefig("plots/interference_vs_kd.png")
    
    # 2. Final Loss vs k/d
    plt.figure(figsize=(10, 6))
    plt.scatter(df_avg['k/d'], df_avg['final_loss'], c='red', label='Avg Loss')
    plt.xlabel('Feature Density (k/d)')
    plt.ylabel('Final Loss (MSE)')
    plt.title('Loss vs Feature Density')
    plt.grid(True)
    plt.savefig("plots/loss_vs_kd.png")
    
    # 3. I/S Ratio vs k/d
    plt.figure(figsize=(10, 6))
    plt.scatter(df_avg['k/d'], df_avg['final_I_S_ratio'], c='green', label='Avg I/S Ratio')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Collapse Threshold (I/S=1)')
    plt.xlabel('Feature Density (k/d)')
    plt.ylabel('I/S Ratio')
    plt.title('Interference/Signal Ratio vs Feature Density')
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/is_ratio_vs_kd.png")
    
    print("Plots saved to 'plots/' directory.")

if __name__ == "__main__":
    main()
