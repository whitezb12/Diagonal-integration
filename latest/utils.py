import warnings
from typing import Optional, List, Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
from anndata import AnnData
from scipy.sparse import issparse, csc_matrix, csr_matrix
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

import anndata
import sklearn

def lsi(adata: anndata.AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs) -> None:
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var

    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)

    if issparse(X):
        X_norm = sklearn.preprocessing.normalize(X, norm="l1")
    else:
        X_norm = sklearn.preprocessing.normalize(X, norm="l1", axis=1)

    X_norm = np.log1p(X_norm * 1e4)

    U, S, _ = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)
    X_lsi = U * S

    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)

    adata.obsm["X_lsi"] = X_lsi


def tfidf(X: np.ndarray) -> np.ndarray:
    if issparse(X):
        row_sums = np.array(X.sum(axis=1)).flatten() + 1e-9
        tf = X.multiply(1 / row_sums[:, np.newaxis])
        idf = np.array(X.shape[0] / X.sum(axis=0)).flatten()
        return tf.multiply(idf)
    else:
        tf = X / (X.sum(axis=1, keepdims=True) + 1e-9)
        idf = X.shape[0] / (X.sum(axis=0) + 1e-9)
        return tf * idf


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


def build_celltype_prior(
    list1: List[Union[str, None]],
    list2: List[Union[str, None]]
) -> torch.Tensor:
    arr1 = np.array(list1, dtype=object)
    arr2 = np.array(list2, dtype=object)

    def missing_mask(arr: np.ndarray) -> np.ndarray:
        arr_str = np.char.lower(np.array(arr, str))
        str_missing = np.isin(arr_str, ["none", "nan", "na", "null", ""])
        none_missing = np.equal(arr, None)
        try:
            nan_missing = np.isnan(arr.astype(float))
        except Exception:
            nan_missing = np.zeros_like(arr, dtype=bool)
        return str_missing | none_missing | nan_missing

    mask1 = missing_mask(arr1)
    mask2 = missing_mask(arr2)

    eq_matrix = np.equal.outer(arr1, arr2).astype(np.float32)

    if mask1.any():
        eq_matrix[mask1, :] = 0
    if mask2.any():
        eq_matrix[:, mask2] = 0

    return torch.from_numpy(eq_matrix)


def pairwise_correlation_distance(
    X: torch.Tensor,
    Y: Optional[torch.Tensor] = None
) -> torch.Tensor:
    Y = X if Y is None else Y
    X_centered = X - X.mean(dim=1, keepdim=True)
    Y_centered = Y - Y.mean(dim=1, keepdim=True)
    cov = X_centered @ Y_centered.T
    std_X = torch.norm(X_centered, p=2, dim=1)
    std_Y = torch.norm(Y_centered, p=2, dim=1)
    corr = cov / (std_X.unsqueeze(1) * std_Y.unsqueeze(0) + 1e-8)
    return 1 - corr


def pairwise_euclidean_distance(
    X: torch.Tensor,
    Y: Optional[torch.Tensor] = None,
    clip: bool = False,
    clip_value: float = 1000.0
) -> torch.Tensor:
    Y = X if Y is None else Y
    if clip:
        max_norm = torch.max(
            torch.abs(X).max() + torch.abs(Y).max(),
            torch.tensor(2 * clip_value, device=X.device, dtype=X.dtype)
        ) / 2
        X, Y = clip_value * X / max_norm, clip_value * Y / max_norm
    X_col, Y_row = X.unsqueeze(1), Y.unsqueeze(0)
    return torch.mean((X_col - Y_row) ** 2, dim=-1)


def pairwise_cosine(
        A: torch.Tensor, 
        B: Optional[torch.Tensor] = None, 
        eps: float = 1e-8):
    B = A if B is None else B
    A_norm = A / (torch.linalg.norm(A, dim=1, keepdim=True) + eps)
    B_norm = B / (torch.linalg.norm(B, dim=1, keepdim=True) + eps)
    return A_norm @ B_norm.T


def unbalanced_ot(
    cost_pp: torch.Tensor,
    reg: float = 0.05,
    reg_m: float = 0.5,
    prior: Optional[torch.Tensor] = None,
    device: str = 'cpu',
    max_iteration: Dict[str, int] = {'outer': 10, 'inner': 5}
) -> Optional[torch.Tensor]:
    ns, nt = cost_pp.shape
    if prior is not None:
        cost_pp = cost_pp * prior

    p_s = torch.ones(ns, 1, device=device) / ns
    p_t = torch.ones(nt, 1, device=device) / nt
    tran = torch.ones(ns, nt, device=device) / (ns * nt)
    dual = torch.ones(ns, 1, device=device) / ns
    f = reg_m / (reg_m + reg)

    for _ in range(max_iteration['outer']):
        cost = cost_pp
        kernel = torch.exp(-cost / (reg * torch.max(torch.abs(cost)))) * tran
        b = p_t / (torch.t(kernel) @ dual)
        for _ in range(max_iteration['inner']):
            dual = (p_s / (kernel @ b)) ** f
            b = (p_t / (torch.t(kernel) @ dual)) ** f
        tran = (dual @ torch.t(b)) * kernel

    out = tran.detach()
    return None if torch.isnan(out).sum() > 0 else out


def Graph_Laplacian_torch(
    X: torch.Tensor,
    nearest_neighbor: int = 30,
    t: float = 1.0
) -> torch.Tensor:
    XX = X.detach()
    D = torch.cdist(XX, XX, p=2)
    values, indices = torch.topk(D, nearest_neighbor + 1, dim=1, largest=False)
    pos = D > values[:, nearest_neighbor].view(-1, 1)
    D[pos] = 0.0
    W = D + D.T.multiply(D.T > D) - D.multiply(D.T > D)
    index_pos = torch.where(W > 0)
    W_mean = torch.mean(W[index_pos])
    W[index_pos] = torch.exp(-W[index_pos] / (t * W_mean))
    return (torch.diag(W.sum(1)) - W).detach()


def Transform(
    X: torch.Tensor,
    Y: torch.Tensor,
    T: torch.Tensor,
    L: torch.Tensor,
    lamda_Eigenvalue: float = 0.0,
    eigenvalue_type: str = 'mean'
) -> torch.Tensor:
    Y, T, L = Y.detach(), T.detach(), L.detach()

    if eigenvalue_type == 'mean':
        a = T.sum(1)
        a_inv = 1.0 / a
        lamda_Lapalcian = 2 * lamda_Eigenvalue / (torch.diag(L) * a_inv).mean()
        l = 2 * torch.mm(T, Y)
        M = lamda_Lapalcian * L + 2 * torch.diag(a)
        M_inv = torch.linalg.inv(M).to(torch.float32) if torch.isnan(M).sum() == 0 else torch.diag(a_inv) / 2.0
        result = torch.mm(M_inv, l)
    elif eigenvalue_type == 'normal':
        lamda_Lapalcian = lamda_Eigenvalue
        l = 2 * torch.mm(T, Y)
        M = lamda_Lapalcian * L + 2 * torch.diag(T.sum(1))
        M_inv = torch.linalg.inv(M).to(torch.float32)
        result = torch.mm(M_inv, l)
    return result


def generalized_clip_loss_stable_masked(
    z_A: torch.Tensor,
    z_B: torch.Tensor,
    Y: torch.Tensor,
    tau: float = 0.1
) -> torch.Tensor:
    mask_A = Y.sum(dim=1) > 0
    mask_B = Y.sum(dim=0) > 0

    z_A_masked = z_A[mask_A]
    z_B_masked = z_B[mask_B]
    Y_masked = Y[mask_A][:, mask_B]

    z_A_masked = z_A_masked / z_A_masked.norm(dim=1, keepdim=True)
    z_B_masked = z_B_masked / z_B_masked.norm(dim=1, keepdim=True)
    S = z_A_masked @ z_B_masked.T / tau

    log_probs_A2B = F.log_softmax(S, dim=1)
    loss_A2B = -(Y_masked * log_probs_A2B).sum(dim=1) / (Y_masked.sum(dim=1) + 1e-8)
    loss_A2B = loss_A2B.mean()

    log_probs_B2A = F.log_softmax(S.T, dim=1)
    loss_B2A = -(Y_masked.T * log_probs_B2A).sum(dim=1) / (Y_masked.T.sum(dim=1) + 1e-8)
    loss_B2A = loss_B2A.mean()

    return (loss_A2B + loss_B2A) / 2


def log_nb_positive(x, mu, theta, eps=1e-8, return_sum=True):
    if theta.ndimension() == 1:
        theta = theta.view(
            1, theta.size(0)
        )  # In this case, we reshape theta for broadcasting

    log_theta_mu_eps = torch.log(theta + mu + eps)

    res = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    if return_sum:
        res = torch.sum(res, dim=-1)
    return res


def zscore_numpy(X: np.ndarray, axis: int = 0, eps: float = 1e-8) -> np.ndarray:
    mean = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, keepdims=True)
    X_z = (X - mean) / (std + eps)
    return X_z


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl.mean()

