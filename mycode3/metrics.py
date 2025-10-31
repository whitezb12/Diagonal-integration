import numpy as np
import scib
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score


def label_transfer(ref, query, embed='X_emb', label_key='celltype', k=5):
    """
    Label transfer using KNN classifier.

    Parameters
    -----------
    ref : AnnData
        Reference AnnData containing the representations and labels.
    query : AnnData
        Query AnnData to transfer labels to.
    embed : str, default='X_emb'
        Key in .obsm specifying which representation to use for classification.
    label_key : str, default='celltype'
        Key in .obs specifying which label to transfer.
    k : int, default=5
        Number of neighbors for KNN classifier.

    Returns
    --------
    np.ndarray
        Predicted labels for the query data.
    """
    X_train = np.asarray(ref.obsm[embed])
    y_train = np.asarray(ref.obs[label_key])
    X_test = np.asarray(query.obsm[embed])

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn.predict(X_test)


def mean_average_precision(adata, embed="X_emb", label_key="celltype", neighbor_frac=0.01, **kwargs):
    """
    Calculate Mean Average Precision (mAP) for embedding quality.
    """
    # Get embedding
    x = np.asarray(adata.obsm[embed]) if embed in adata.obsm and adata.obsm[embed] is not None else np.asarray(adata.X)

    # Get labels
    if label_key not in adata.obs:
        raise KeyError(f"Label key '{label_key}' not found in adata.obs.")
    y = np.asarray(adata.obs[label_key])

    n_samples = y.shape[0]
    k = max(round(n_samples * neighbor_frac), 1)

    # Nearest neighbor search
    nn = NearestNeighbors(n_neighbors=min(n_samples, k + 1), **kwargs).fit(x)
    nni = nn.kneighbors(x, return_distance=False)[:, 1:]  # exclude self

    # Matching labels
    match = np.equal(y[nni], np.expand_dims(y, 1))

    # Compute mean average precision per sample
    def _average_precision(row):
        if np.any(row):
            cummean = np.cumsum(row) / (np.arange(len(row)) + 1)
            return cummean[row].mean()
        return 0.0

    map_score = np.apply_along_axis(_average_precision, 1, match).mean()
    return float(map_score)


def run_metrics(
    adata,
    batch_key,
    label_key,
    embed="X_pca",
    cluster_key="cluster",
    nmi_method="arithmetic",
    nmi_dir=None,
    si_metric="euclidean",
    subsample=0.5,
    n_cores=1,
    type_=None,
    verbose=False,
):
    """
    Run a suite of integration and clustering metrics using scIB.

    Returns
    -------
    dict
        Metric results.
    """

    # ===== Biological conservation metrics =====
    map_score = mean_average_precision(adata, embed=embed, label_key=label_key)
    asw_label = scib.metrics.silhouette(adata, label_key=label_key, embed=embed, metric=si_metric)

    # ===== Clustering metrics =====
    _, _, _ = scib.metrics.cluster_optimal_resolution(
        adata, label_key=label_key, cluster_key=cluster_key, use_rep=embed, force=True,
        verbose=verbose, return_all=True
    )

    nmi_score = scib.metrics.nmi(
        adata, cluster_key=cluster_key, label_key=label_key,
        implementation=nmi_method, nmi_dir=nmi_dir
    )
    ari_score = float(scib.metrics.ari(adata, cluster_key=cluster_key, label_key=label_key))
    clisi = scib.metrics.clisi_graph(
        adata, batch_key=batch_key, label_key=label_key, type_=type_,
        subsample=subsample * 100, scale=True, n_cores=n_cores, verbose=verbose
    )

    # ===== Batch correction metrics =====
    asw_batch = scib.metrics.silhouette_batch(
        adata, batch_key=batch_key, label_key=label_key, embed=embed, metric=si_metric
    )
    graph_conn = scib.metrics.graph_connectivity(adata, label_key=label_key)
    kbet = scib.metrics.kBET(
        adata, batch_key=batch_key, label_key=label_key, type_=type_,
        embed=embed, scaled=True, verbose=verbose
    )
    ilisi = scib.metrics.ilisi_graph(
        adata, batch_key=batch_key, type_=type_,
        subsample=subsample * 100, scale=True, n_cores=n_cores, verbose=verbose
    )

    # ===== Classification (label transfer) =====
    batches = adata.obs[batch_key].unique()
    if len(batches) != 2:
        raise ValueError("label_transfer currently supports exactly two batches.")
    adata_A = adata[adata.obs[batch_key] == batches[0]]
    adata_B = adata[adata.obs[batch_key] == batches[1]]

    y_A, y_B = adata_A.obs[label_key], adata_B.obs[label_key]
    pred_A = label_transfer(adata_B, adata_A, embed=embed, label_key=label_key)
    pred_B = label_transfer(adata_A, adata_B, embed=embed, label_key=label_key)

    transfer_acc = (accuracy_score(y_A, pred_A) + accuracy_score(y_B, pred_B)) / 2
    transfer_f1 = (f1_score(y_A, pred_A, average='micro') + f1_score(y_B, pred_B, average='micro')) / 2

    # ===== Consolidate results =====
    return {
        "Mean_average_precision": map_score,
        "ASW_label": asw_label,
        "NMI_cluster/label": nmi_score,
        "ARI_cluster/label": ari_score,
        "cLISI": clisi,
        "ASW_batch": asw_batch,
        "Graph_connectivity": graph_conn,
        "iLISI": ilisi,
        "KBET": kbet,
        "transfer_accuracy": transfer_acc,
        "transfer_f1": transfer_f1,
    }
