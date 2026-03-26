import torch
import torch.nn as nn
import numpy as np

def sinkhorn_topk(x, k, beta=0.1, n_iters=10):
    """
    Entropy-regularized Soft-TopK via Sinkhorn Iterations.
    """
    b, n = x.shape
    C = x / (beta + 1e-9)
    
    mask = torch.sigmoid(C)
    for _ in range(n_iters):
        scaling = k / (torch.sum(mask, dim=1, keepdim=True) + 1e-9)
        mask = torch.sigmoid(C * scaling)
        
    return x * mask

class FeatureDecoder(nn.Module):
    """
    Upgraded Decoder with Weight Normalization and Temperature Scaling.
    Addresses "Low-Energy Weight Collapse" (Research Note 4).
    """
    def __init__(self, feature_dim, num_features, top_k=None, threshold=None, iterative_steps=0, beta=0.1, tau=1.0):
        super(FeatureDecoder, self).__init__()
        self.feature_dim = feature_dim
        self.num_features = num_features
        self.top_k = top_k
        self.threshold = threshold
        self.iterative_steps = iterative_steps
        self.beta = beta 
        self.tau = tau # Sigmoid temperature (Inquiry 3)
        
        # Initializing W on the unit sphere
        weights = torch.randn(num_features, feature_dim)
        weights = weights / (torch.norm(weights, dim=1, keepdim=True) + 1e-9)
        self.W = nn.Parameter(weights)
        self.sigmoid = nn.Sigmoid()

    def get_normalized_W(self):
        # Force Weight-Normalization Constraint (Inquiry 1)
        return self.W / (torch.norm(self.W, dim=1, keepdim=True) + 1e-9)

    def forward(self, h):
        # 1. Temperature-Scaled Sigmoid to maintain Dynamic Isometry
        # Addresses Inquiry 3: Saturated Sigmoid damping
        activated = self.sigmoid(h / self.tau)
        
        # 2. Use normalized weights to prevent energy collapse
        W_norm = self.get_normalized_W()
        y = torch.matmul(activated, W_norm.t())
        
        # Iterative Refinement
        for _ in range(self.iterative_steps):
            support = (y > (self.threshold if self.threshold else 0.5)).float()
            y = y * (0.5 + 0.5 * support) 

        # Differentiable Sparsity Constraint
        if self.top_k is not None:
            if self.top_k < self.num_features:
                y = sinkhorn_topk(y, self.top_k, beta=self.beta)
        
        if not self.training and self.threshold is not None:
            y = torch.where(y >= self.threshold, y, torch.zeros_like(y))
                
        return y

def calculate_interference_metrics(W, F):
    """
    Calculates metrics with support for normalized weights.
    """
    num_features = W.shape[0]
    device = W.device
    
    # Ensure W is normalized for metric consistency
    W = W / (torch.norm(W, dim=1, keepdim=True) + 1e-9)
    
    dot_products = torch.abs(torch.matmul(W, F.t()))
    signals = torch.diagonal(dot_products)
    S = torch.mean(signals).item()
    
    mask = torch.eye(num_features, device=device).bool()
    interference_per_feature = dot_products * (~mask).float()
    I = torch.mean(interference_per_feature).item()
    
    interference_sums = torch.sum(interference_per_feature, dim=1, keepdim=True) + 1e-9
    I_normalized = interference_per_feature / interference_sums
    entropy_per_feature = -torch.sum(I_normalized * torch.log(I_normalized + 1e-9), dim=1)
    H_I = torch.mean(entropy_per_feature).item()
    
    FtF = torch.matmul(F.t(), F)
    try:
        s = torch.linalg.svdvals(FtF)
        cond_number = (s[0] / s[-1]).item() if s[-1] > 1e-9 else float('inf')
    except Exception:
        cond_number = float('nan')
        
    F_norm = F / (torch.norm(F, dim=1, keepdim=True) + 1e-9)
    Gram = torch.matmul(F_norm, F_norm.t())
    identity = torch.eye(num_features, device=device)
    ortho_error = torch.norm(Gram - identity).item()
    
    return {
        "signal": S,
        "interference": I,
        "I_S_ratio": I / S if S > 0 else float('inf'),
        "interference_entropy": H_I,
        "condition_number": cond_number,
        "ortho_error": ortho_error
    }
