import numpy as np
import scanpy as sc
from anndata import AnnData
from typing import Tuple, Union


def construct_graph(
    adata1: AnnData,
    adata2: AnnData,
    n_neighbors: int = 15,
    n_pcs: int = 30,
    metric: str = 'correlation'
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:

    sc.pp.pca(adata1, n_comps=n_pcs)
    sc.pp.pca(adata2, n_comps=n_pcs)

    sc.pp.neighbors(adata1, n_neighbors=n_neighbors, use_rep='X_pca', metric=metric)
    rows1, cols1 = adata1.obsp['connectivities'].nonzero()
    vals1 = adata1.obsp['connectivities'][(rows1, cols1)].A1
    edges1 = (rows1, cols1, vals1)

    sc.pp.neighbors(adata2, n_neighbors=n_neighbors, use_rep='X_pca', metric=metric)
    rows2, cols2 = adata2.obsp['connectivities'].nonzero()
    vals2 = adata2.obsp['connectivities'][(rows2, cols2)].A1
    edges2 = (rows2, cols2, vals2)

    return edges1, edges2


def graph_smoothing(
    arr: np.ndarray,
    edges: Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    wt: float
) -> np.ndarray:

    n_samples, n_features = arr.shape
    src = np.asarray(edges[0])
    tgt = np.asarray(edges[1])
    w = np.asarray(edges[2]) if len(edges) == 3 else np.ones_like(src, dtype=np.float32)

    smoothed = np.zeros_like(arr)
    total_weight = np.zeros((n_samples, 1), dtype=np.float32)

    np.add.at(smoothed, src, arr[tgt] * w[:, None])
    np.add.at(total_weight, src, w[:, None])

    centroids = arr.copy()
    mask = total_weight.squeeze() > 0
    centroids[mask] = smoothed[mask] / total_weight[mask]

    return wt * arr + (1.0 - wt) * centroids


def smooth_link_feat(
    adata1: AnnData,
    adata2: AnnData,
    n_neighbors: int = 15,
    metric: str = 'correlation',
    weight: float = 0.3,
    n_pcs: int = 30
) -> None:

    if 'link_feat' not in adata1.obsm or 'link_feat' not in adata2.obsm:
        raise ValueError("Missing 'link_feat' in obsm of one or both AnnData objects.")

    edges1, edges2 = construct_graph(
        adata1=adata1,
        adata2=adata2,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        metric=metric
    )

    smoothed1 = graph_smoothing(adata1.obsm['link_feat'], edges1, wt=weight)
    smoothed2 = graph_smoothing(adata2.obsm['link_feat'], edges2, wt=weight)

    adata1.obsm['link_feat'] = smoothed1
    adata2.obsm['link_feat'] = smoothed2
