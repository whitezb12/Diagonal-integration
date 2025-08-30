import numpy as np
import torch
from typing import Optional, List, Dict
from anndata import AnnData
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from scipy.sparse import issparse, csc_matrix, csr_matrix
import warnings


def clr(adata: AnnData, axis: int = 0) -> None:
    if axis not in [0, 1]:
        raise ValueError("axis must be 0 or 1")

    if issparse(adata.X) and axis == 0 and not isinstance(adata.X, csc_matrix):
        warnings.warn("adata.X is sparse but not in CSC format. Converting to CSC.")
        x = csc_matrix(adata.X)
    elif issparse(adata.X) and axis == 1 and not isinstance(adata.X, csr_matrix):
        warnings.warn("adata.X is sparse but not in CSR format. Converting to CSR.")
        x = csr_matrix(adata.X)
    else:
        x = adata.X

    if issparse(x):
        x.data /= np.repeat(
            np.exp(np.log1p(x).sum(axis=axis).A / x.shape[axis]),
            x.getnnz(axis=axis)
        )
        np.log1p(x.data, out=x.data)
    else:
        np.log1p(
            x / np.exp(np.log1p(x).sum(axis=axis, keepdims=True) / x.shape[axis]),
            out=x,
        )

    adata.X = x


def batch_scale(adata: AnnData, method: str = 'maxabs') -> None:
    if 'batch' in adata.obs:
        batches = adata.obs['batch'].unique()
    else:
        print("No 'batch' found in adata.obs, applying scaling to all data.")
        batches = [None]

    for b in batches:
        if b is None:
            idx = np.arange(adata.n_obs)
        else:
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


def build_celltype_prior(list1: List[str], list2: List[str], prior: float = 2.0) -> torch.Tensor:
    arr1 = np.array(list1)
    arr2 = np.array(list2)
    is_same_tensor = torch.from_numpy(arr1[:, None] == arr2)
    return torch.where(is_same_tensor, 1.0 / prior, prior)


def pairwise_correlation_distance(X: torch.Tensor, Y: Optional[torch.Tensor] = None) -> torch.Tensor:
    if Y is None:
        Y = X
    X_centered = X - X.mean(dim=1, keepdim=True)
    Y_centered = Y - Y.mean(dim=1, keepdim=True)
    cov = X_centered @ Y_centered.T
    std_X = torch.norm(X_centered, p=2, dim=1)
    std_Y = torch.norm(Y_centered, p=2, dim=1)
    corr = cov / (std_X.unsqueeze(1) * std_Y.unsqueeze(0) + 1e-8)
    return 1 - corr

def pairwise_euclidean_distance(X: torch.Tensor, Y: Optional[torch.Tensor] = None) -> torch.Tensor:
    if Y is None:
        Y = X
    return ((X[:, None, :] - Y[None, :, :]) ** 2).sum(dim=-1)

def unbalanced_ot(cost_pp: torch.Tensor, 
                  reg: float = 0.05, 
                  reg_m: float = 0.5, 
                  prior: Optional[torch.Tensor] = None, 
                  device: str = 'cpu', 
                  max_iteration: Dict[str, int] = {'outer': 10, 'inner': 5}) -> Optional[torch.Tensor]:
    outer_iter = max_iteration['outer']
    inner_iter = max_iteration['inner']
    ns, nt = cost_pp.shape
    if prior is not None:
        cost_pp = cost_pp * prior
    p_s = torch.ones(ns, 1, device=device) / ns   
    p_t = torch.ones(nt, 1, device=device) / nt
    tran = torch.ones(ns, nt, device=device) / (ns * nt)
    dual = torch.ones(ns, 1, device=device) / ns
    f = reg_m / (reg_m + reg)
    for _ in range(outer_iter):
        cost = cost_pp
        kernel = torch.exp(-cost / (reg * torch.max(torch.abs(cost)))) * tran
        b = p_t / (torch.t(kernel) @ dual)
        for _ in range(inner_iter):
            dual = (p_s / (kernel @ b))**f
            b = (p_t / (torch.t(kernel) @ dual))**f
        tran = (dual @ torch.t(b)) * kernel
    out = tran.detach()
    if torch.isnan(out).sum() > 0:
        return None
    return out


def Graph_topk(X: torch.Tensor, nearest_neighbor: int = 10, t: float = 1.0) -> torch.Tensor:
    XX = X.detach()
    D = pairwise_correlation_distance(XX)
    values, _ = torch.topk(D, nearest_neighbor + 1, dim=1, largest=False)
    radius = values[:, nearest_neighbor].view(-1, 1)
    W = torch.where(D <= radius, D, torch.zeros_like(D))
    W = torch.max(W, W.T)
    pos = W > 0
    W_mean = W[pos].mean()
    W[pos] = torch.exp(-W[pos] / (t * W_mean))
    return W.detach()
