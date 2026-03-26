import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

def calculate_entropy(scores):
    # Normalized scores for entropy
    probs = torch.abs(scores) / (torch.sum(torch.abs(scores)) + 1e-9)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9))
    return entropy.item()

def analyze_attention_interference(seq_len, d_model, num_heads):
    """
    Simulates a Multi-Head Attention layer and measures interference between tokens.
    """
    query_layer = nn.Linear(d_model, d_model)
    key_layer = nn.Linear(d_model, d_model)
    
    # Random input (batch_size=1)
    x = torch.randn(1, seq_len, d_model)
    
    Q = query_layer(x) # (1, seq_len, d_model)
    K = key_layer(x)   # (1, seq_len, d_model)
    
    # Reshape for multi-head
    Q = Q.view(1, seq_len, num_heads, d_model // num_heads).transpose(1, 2)
    K = K.view(1, seq_len, num_heads, d_model // num_heads).transpose(1, 2)
    
    # QK^T (1, num_heads, seq_len, seq_len)
    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_model // num_heads)**0.5
    
    # Interference metric in attention:
    diag = torch.diagonal(attention_scores, dim1=-2, dim2=-1)
    S_att = torch.mean(torch.abs(diag)).item()
    
    mask = torch.eye(seq_len).bool()
    off_diag = attention_scores[:, :, ~mask]
    I_att = torch.mean(torch.abs(off_diag)).item()
    
    # Advanced: Interference Entropy H_I (Question 5)
    H_I = calculate_entropy(off_diag)
    
    return S_att, I_att, I_att / S_att if S_att > 0 else float('inf'), H_I

def main():
    seq_len = 32
    d_model = 128
    heads = [1, 2, 4, 8, 16]
    
    results = []
    for h in heads:
        S, I, ratio, entropy = analyze_attention_interference(seq_len, d_model, h)
        results.append({"heads": h, "S": S, "I": I, "ratio": ratio, "entropy": entropy})
        
    print("--- Transformer Attention Interference Analysis ---")
    print(f"Seq Len: {seq_len}, Model Dim: {d_model}")
    for res in results:
        print(f"Heads: {res['heads']:2d} | Signal: {res['S']:.4f} | Interference: {res['I']:.4f} | Ratio: {res['ratio']:.4f} | Entropy: {res['entropy']:.4f}")

if __name__ == "__main__":
    main()
