import numpy as np
import scib
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score


def label_transfer(ref, query, embed="X_emb", label_key="celltype", k=5, metric="euclidean"):
    X_train = np.asarray(ref.obsm[embed]) if embed in ref.obsm else np.asarray(ref.X)
    X_test = np.asarray(query.obsm[embed]) if embed in query.obsm else np.asarray(query.X)
    y_train = np.asarray(ref.obs[label_key])
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    knn.fit(X_train, y_train)
    return knn.predict(X_test)


def mean_average_precision(adata, embed="X_emb", label_key="celltype", neighbor_frac=0.01, **kwargs):
    x = np.asarray(adata.obsm[embed]) if embed in adata.obsm and adata.obsm[embed] is not None else np.asarray(adata.X)
    if label_key not in adata.obs:
        raise KeyError(f"Label key '{label_key}' not found in adata.obs.")
    y = np.asarray(adata.obs[label_key])
    n_samples = y.shape[0]
    k = max(round(n_samples * neighbor_frac), 1)
    nn = NearestNeighbors(n_neighbors=min(n_samples, k + 1), **kwargs).fit(x)
    nni = nn.kneighbors(x, return_distance=False)[:, 1:]
    match = np.equal(y[nni], np.expand_dims(y, 1))

    def _average_precision(row):
        if np.any(row):
            cummean = np.cumsum(row) / (np.arange(len(row)) + 1)
            return cummean[row].mean()
        return 0.0

    map_score = np.apply_along_axis(_average_precision, 1, match).mean()
    return float(map_score)