import numpy as np
import torch
import scanpy as sc
import anndata as ad
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

def batch_scale(adata, method='maxabs'):
    for b in adata.obs['batch'].unique():
        idx = np.where(adata.obs['batch'] == b)[0]
        if method == 'maxabs':
            scaler = MaxAbsScaler(copy=False).fit(adata.X[idx])
        elif method == 'standard':
            scaler = StandardScaler(copy=False).fit(adata.X[idx])
        else:
            raise ValueError(f"Unknown scaling method: {method}. Choose 'maxabs' or 'standard'.")
        adata.X[idx] = scaler.transform(adata.X[idx])

   
def build_mnn_prior(Sim: torch.Tensor, k: int, prior: float = 2.0) -> torch.Tensor:
    row_mask = torch.zeros_like(Sim, dtype=torch.bool)
    col_mask = torch.zeros_like(Sim, dtype=torch.bool)
    row_mask.scatter_(1, Sim.topk(k, dim=1, largest=False).indices, True)
    col_mask.scatter_(0, Sim.topk(k, dim=0, largest=False).indices, True)
    mnn_mask = row_mask & col_mask
    return torch.where(mnn_mask, 1.0/prior, prior)

def build_celltype_prior(list1, list2, prior: float = 2.0) -> torch.Tensor:
    arr1 = np.array(list1)
    arr2 = np.array(list2)
    is_same = arr1[:, None] == arr2
    is_same_tensor = torch.from_numpy(is_same)
    return torch.where(is_same_tensor, 1.0 / prior, prior)




def pairwise_correlation_distance(X, Y=None):
    if Y is None:
        Y = X
    X_centered = X - X.mean(dim=1, keepdim=True)
    Y_centered = Y - Y.mean(dim=1, keepdim=True)
    cov = X_centered @ Y_centered.T
    std_X = torch.norm(X_centered, p=2, dim=1)
    std_Y = torch.norm(Y_centered, p=2, dim=1)
    std_prod = std_X.unsqueeze(1) * std_Y.unsqueeze(0)
    corr = cov / (std_prod + 1e-8)
    return 1 - corr

