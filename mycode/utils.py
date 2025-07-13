import numpy as np
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from anndata import AnnData
import torch
from scipy.sparse import issparse, csc_matrix, csr_matrix
import warnings

def clr(adata: AnnData, axis: int = 0) -> None:
    if axis not in [0, 1]:
        raise ValueError("axis must be 0 or 1")

    if issparse(adata.X) and axis == 0 and not isinstance(adata.X, csc_matrix):
        warnings("adata.X is sparse but not in CSC format. Converting to CSC.")
        x = csc_matrix(adata.X)
    elif issparse(adata.X) and axis == 1 and not isinstance(adata.X, csr_matrix):
        warnings("adata.X is sparse but not in CSR format. Converting to CSR.")
        x = csr_matrix(adata.X)
    else:
        x = adata.X

    if issparse(x):
        x.data /= np.repeat(
            np.exp(np.log1p(x).sum(axis=axis).A / x.shape[axis]), x.getnnz(axis=axis)
        )
        np.log1p(x.data, out=x.data)
    else:
        np.log1p(
            x / np.exp(np.log1p(x).sum(axis=axis, keepdims=True) / x.shape[axis]),
            out=x,
        )

    adata.X = x


def batch_scale(adata: AnnData, method: str = 'maxabs') -> None:
    from scipy.sparse import issparse
    from sklearn.preprocessing import MaxAbsScaler, StandardScaler

    for b in adata.obs['batch'].unique():
        idx = np.where(adata.obs['batch'] == b)[0]
        X_batch = adata.X[idx]

        if issparse(X_batch):
            if method == 'standard':
                scaler = StandardScaler(with_mean=False, copy=False).fit(X_batch)
                adata.X[idx] = scaler.transform(X_batch)
            elif method == 'maxabs':
                scaler = MaxAbsScaler(copy=False).fit(X_batch)
                adata.X[idx] = scaler.transform(X_batch)
            else:
                raise ValueError(f"Unknown scaling method: {method}. Choose 'maxabs' or 'standard'.")
        else:
            if method == 'standard':
                scaler = StandardScaler(copy=False).fit(X_batch)
            elif method == 'maxabs':
                scaler = MaxAbsScaler(copy=False).fit(X_batch)
            else:
                raise ValueError(f"Unknown scaling method: {method}. Choose 'maxabs' or 'standard'.")

            adata.X[idx] = scaler.transform(X_batch)


def build_mnn_prior(Sim: torch.Tensor, k: int, prior: float = 2.0) -> torch.Tensor:
    row_mask = torch.zeros_like(Sim, dtype=torch.bool)
    col_mask = torch.zeros_like(Sim, dtype=torch.bool)
    row_mask.scatter_(1, Sim.topk(k, dim=1, largest=False).indices, True)
    col_mask.scatter_(0, Sim.topk(k, dim=0, largest=False).indices, True)
    mnn_mask = row_mask & col_mask
    return torch.where(mnn_mask, 1.0 / prior, prior)


def build_celltype_prior(list1, list2, prior: float = 2.0) -> torch.Tensor:
    arr1 = np.array(list1)
    arr2 = np.array(list2)
    is_same = arr1[:, None] == arr2
    is_same_tensor = torch.from_numpy(is_same)
    return torch.where(is_same_tensor, 1.0 / prior, prior)


def pairwise_correlation_distance(X: torch.Tensor, Y: torch.Tensor = None) -> torch.Tensor:
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
