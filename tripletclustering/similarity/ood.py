import numpy as np
import pynndescent
import scipy.sparse as sp
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_array


def knn_rank_graph(X, n_neighbors=15, query_neighbors=None, **kwargs) -> sp.csr_array:
    if sp.issparse(X):
        # TODO remove once pynndescent allows sparse arrays
        X = sp.csr_matrix(X)
    n = X.shape[0]
    if query_neighbors is None:
        query_neighbors = min(5, n_neighbors // 5)
    index = pynndescent.NNDescent(X, n_neighbors=query_neighbors, **kwargs)
    # Get one more neighbor than n_neighbors so we can drop the point itself
    neighbors = index.query(X, k=n_neighbors + 1)[0]
    neighbors = neighbors[:, 1:]  # Drop the point itself
    ranks = np.tile(np.arange(n_neighbors, dtype="int32") + 1, (n, 1))
    # Sort by neighbor index
    sort_index = np.argsort(neighbors)
    neighbors = np.take_along_axis(neighbors, sort_index)
    ranks = np.take_along_axis(ranks, sort_index)
    # Make ood as csr array
    indptr = np.arange(n + 1, dtype="int32") * n_neighbors
    indices = neighbors.reshape(-1)
    data = neighbors.reshape(-1)
    ood = sp.csr_array((data, indices, indptr), shape=(n, n))
    return ood


def full_ood(X, metric="cosine"):
    X = check_array(X, accept_sparse=True)
    D = pairwise_distances(X, metric=metric)
    return D


# TODO numbaify
def prune_to_knn(X, n_neighbors=15):
    if n_neighbors > X.shape[0]:
        raise ValueError(
            f"Cannot prune to {n_neighbors} nearest neighbors when only {X.shape[0]} objects exist."
        )
    X = sp.csr_array(X)
    for row in range(X.shape[0]):
        row_neighbors = X.indices[X.indptr[row] : X.indptr[row + 1]]
        row_data = X.data[X.indptr[row] : X.indptr[row + 1]]
        if len(row_neighbors) <= n_neighbors:
            continue
        argsort = np.argsort(row_data)
        if row_neighbors[argsort[0]] == row:
            offset = X.indptr[row]
            remove_indices = argsort[n_neighbors + 1 :] + offset
            X.data[remove_indices] = 0
            X.data[argsort[0] + offset] = 0
        else:
            offset = X.indptr[row]
            remove_indices = argsort[n_neighbors:] + offset
            X.data[remove_indices] = 0
    X.eliminate_zeros()
    return X
