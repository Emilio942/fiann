import torch
from torch.utils.data import Dataset, DataLoader

class SyntheticFeatureDataset(Dataset):
    def __init__(self, num_features, feature_dim, num_samples, sparsity=0.1, seed=42):
        """
        num_features (k): Number of possible features
        feature_dim (d): Dimension of the hidden space
        num_samples: Number of samples in the dataset
        sparsity: Fraction of features active in each sample
        """
        torch.manual_seed(seed)
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.num_samples = num_samples
        self.sparsity = sparsity
        
        # 1. Feature Vectors F (k, d)
        # Randomly initialized on the unit sphere
        self.F = torch.randn(num_features, feature_dim)
        self.F = self.F / torch.norm(self.F, dim=1, keepdim=True)
        
        # 2. Sparse Activations A (num_samples, k)
        # We use a Bernoulli distribution for which features are active
        # and a uniform or normal distribution for the values.
        active_mask = torch.rand(num_samples, num_features) < sparsity
        self.A = torch.rand(num_samples, num_features) * active_mask.float()
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        a = self.A[idx] # (k,)
        # h = sum(a_i * f_i)
        h = torch.matmul(a, self.F) # (d,)
        return h, a

def get_dataloader(num_features, feature_dim, num_samples, sparsity=0.1, batch_size=32, seed=42):
    dataset = SyntheticFeatureDataset(num_features, feature_dim, num_samples, sparsity, seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), dataset.F
