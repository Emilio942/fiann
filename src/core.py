import torch
import torch.nn as nn
import numpy as np

class FeatureDecoder(nn.Module):
    """
    Implements the model: y = W * sigmoid(h), where h = sum(a_i * f_i)
    In this simplified version, we assume the input h is already the mixture of features.
    """
    def __init__(self, feature_dim, num_features, top_k=None):
        super(FeatureDecoder, self).__init__()
        self.feature_dim = feature_dim
        self.num_features = num_features
        self.top_k = top_k
        # W has shape (num_features, feature_dim)
        self.W = nn.Parameter(torch.randn(num_features, feature_dim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        """
        h: Mixture of features (batch_size, feature_dim)
        returns: y (batch_size, num_features)
        """
        # Applying sigmoid as per task 1.2
        activated = self.sigmoid(h)
        # Linear decoding
        y = torch.matmul(activated, self.W.t())
        
        # Top-k sparsity if enabled
        if self.top_k is not None:
            # Keep only the top k activations, set others to 0
            if self.top_k < self.num_features:
                top_values, _ = torch.topk(y, self.top_k, dim=1)
                min_top = top_values[:, -1].unsqueeze(1)
                y = torch.where(y >= min_top, y, torch.zeros_like(y))
                
        return y

def calculate_interference_metrics(W, F):
    """
    Calculates metrics defined in Task 1.3
    W: Decoder weights (num_features, feature_dim)
    F: Feature matrix (num_features, feature_dim) - the actual feature vectors f_i
    """
    num_features = W.shape[0]
    
    # 1. Signal vs. Interference
    # S = E[|w_i^T f_i|]
    signals = torch.abs(torch.sum(W * F, dim=1))
    S = torch.mean(signals).item()
    
    # I = E_{i!=j}[|w_i^T f_j|]
    # Compute all-to-all dot products: (num_features, num_features)
    dot_products = torch.abs(torch.matmul(W, F.t()))
    # Mask out the diagonal (i=j)
    mask = torch.eye(num_features, device=W.device).bool()
    interference_values = dot_products[~mask]
    I = torch.mean(interference_values).item()
    
    # 2. Condition Number κ(F^T F)
    # F^T F has shape (feature_dim, feature_dim)
    FtF = torch.matmul(F.t(), F)
    try:
        # Use singular values for condition number: sigma_max / sigma_min
        s = torch.linalg.svdvals(FtF)
        cond_number = (s[0] / s[-1]).item() if s[-1] > 1e-9 else float('inf')
    except Exception:
        cond_number = float('nan')
        
    # 3. Orthogonality Error |F^T F - I|
    # Note: If F has num_features > feature_dim, F^T F is dxd, but we usually look at FF^T (kxk)
    # for feature orthogonality if k > d. Task says κ(F^T F), so we stick to that for cond.
    # For orthogonality error, we look at the deviation of the Gram matrix from identity.
    # Assuming normalized features for this metric to be meaningful.
    F_norm = F / (torch.norm(F, dim=1, keepdim=True) + 1e-9)
    Gram = torch.matmul(F_norm, F_norm.t())
    identity = torch.eye(num_features, device=W.device)
    ortho_error = torch.norm(Gram - identity).item()
    
    return {
        "signal": S,
        "interference": I,
        "I_S_ratio": I / S if S > 0 else float('inf'),
        "condition_number": cond_number,
        "ortho_error": ortho_error
    }
