import numpy as np
import scanpy as sc

def construct_graph(adata1, adata2, n_neighbors=15, metric='correlation'):
    sc.pp.neighbors(adata1, n_neighbors=n_neighbors, n_pcs=None, use_rep='X', metric=metric)
    rows1, cols1 = adata1.obsp['connectivities'].nonzero()
    vals1 = adata1.obsp['connectivities'][(rows1, cols1)].A1
    
    sc.pp.neighbors(adata2, n_neighbors=n_neighbors, n_pcs=None, use_rep='X', metric=metric)
    rows2, cols2 = adata2.obsp['connectivities'].nonzero()
    vals2 = adata2.obsp['connectivities'][(rows2, cols2)].A1

    edges1 = (rows1, cols1, vals1)
    edges2 = (rows2, cols2, vals2)
    return edges1, edges2

def graph_smoothing(arr, edges, wt):
    n_samples, n_features = arr.shape
    src = np.asarray(edges[0])
    tgt = np.asarray(edges[1])

    if len(edges) == 3:
        w = np.asarray(edges[2])
    else:
        w = np.ones_like(src, dtype=np.float32)

    smoothed = np.zeros_like(arr)
    total_weight = np.zeros((n_samples, 1), dtype=np.float32)

    np.add.at(smoothed, src, arr[tgt] * w[:, None])
    np.add.at(total_weight, src, w[:, None])

    mask = total_weight.squeeze() > 0
    centroids = np.copy(arr)
    centroids[mask] = smoothed[mask] / total_weight[mask]

    return wt * arr + (1 - wt) * centroids

def smooth_link_feat(adata1, adata2, n_neighbors=15, metric='correlation', weight=0.3):
    link_feat1 = adata1.obsm['link_feat']
    link_feat2 = adata2.obsm['link_feat']

    edges1, edges2 = construct_graph(adata1, adata2, n_neighbors=n_neighbors, metric=metric)

    smooth_feat1 = graph_smoothing(arr=link_feat1, edges=edges1, wt=weight)
    smooth_feat2 = graph_smoothing(arr=link_feat2, edges=edges2, wt=weight)

    adata1.obsm['link_feat'] = smooth_feat1
    adata2.obsm['link_feat'] = smooth_feat2
