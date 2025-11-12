import numpy as np
import sklearn.neighbors

def mean_average_precision(
    adata,
    embed: str = "X_emb",
    label_key: str = "celltype",
    neighbor_frac: float = 0.01,
    **kwargs,
) -> float:
    if embed in adata.obsm and adata.obsm[embed] is not None:
        x = np.asarray(adata.obsm[embed])
    else:
        x = np.asarray(adata.X)
    if label_key not in adata.obs:
        raise KeyError(f"Label key '{label_key}' not found in adata.obs.")
    y = np.asarray(adata.obs[label_key])

    n_samples = y.shape[0]
    k = max(round(n_samples * neighbor_frac), 1)
    nn = sklearn.neighbors.NearestNeighbors(
        n_neighbors=min(n_samples, k + 1), **kwargs
    ).fit(x)
    nni = nn.kneighbors(x, return_distance=False)
    match = np.equal(y[nni[:, 1:]], np.expand_dims(y, 1))

    def _average_precision(match_row: np.ndarray) -> float:
        if np.any(match_row):
            cummean = np.cumsum(match_row) / (np.arange(match_row.size) + 1)
            return cummean[match_row].mean()
        return 0.0

    return float(np.apply_along_axis(_average_precision, 1, match).mean())
