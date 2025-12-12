import scipy.sparse as sp
import numpy as np
import pynndescent
from numba import njit
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array
import warnings

from .clustering_utils import hslc, prune_clusters

@njit
def size_of_union(a, b):
    return len(a) + len(b) - len(np.intersect1d(a, b))


@njit  # Already optimal parallelized
def make_other_knn_uxy_sizes(knn):
    other_sizes = np.empty_like(knn)
    cache = dict()
    for i in range(knn.shape[0]):
        for j in range(knn.shape[1]):
            if (i, j) in cache:
                other_sizes[i, j] = cache[(i, j)]
            else:
                size = size_of_union(knn[i, :], knn[knn[i, j], :])
                other_sizes[i, j] = size
                cache[(j, i)] = size
    return other_sizes


@njit
def make_n_closer_than(knn, knn_dist):
    this_dist = np.empty(
        (knn.shape[0], knn.shape[1], knn.shape[1]), dtype=knn_dist.dtype
    )
    for i in range(knn.shape[1]):
        this_dist[:, :, i] = knn_dist

    other_dist = np.empty_like(this_dist)
    for i in range(knn.shape[0]):
        other_dist[i, :, :] = knn_dist[knn[i, :], :]

    n_closer_than = np.sum(this_dist < other_dist, axis=2) + np.sum(
        this_dist == other_dist, axis=2
    )
    return n_closer_than


class PAKNNLD(ClusterMixin, BaseEstimator):
    """
    Partitioned K Nearest Neighbors Local Depth clustering algorithm. Based on the assertion that
    clustering is primarily a local problem, and uses on K nearest neighbors to compute cohesion
    values. Significantly faster than PALD, something like O(k^2 * n * log n) and suitable for hundreds
    thousands of points. Uses the Leiden clustering algorithm with modularity to cluster the cohesion
    graph (different than either paper).

    Extension of the paper "A social perspective
    on perceived distances reveals deep community structure" by Kenneth Berenhaut, Katherine
    Moorea, and Ryan Melvin,
    https://www.pnas.org/doi/10.1073/pnas.2003634119
    and mentioned in the paper "Partitioned K-nearest neighbor local depth for scalable
    comparison-based learning" by Baron, Darling, Davis, and Pettit.
    https://arxiv.org/abs/2108.08864

    Parameters
    ----------

    n_neighbors : int, default=100

    metric : string, default="cosine"
        Passed to pynndescent metric so a wide range of options are supported.

    threshold : float or string
        To cut down the cohesion graph for clustering. If thresh is a float remove
        the bottom thresh percentile of weights from the graph.

    cluster_selection_method: string, default="cc"
        Specify the method for selecting clusters from the pruned symmetric cohesion graph.
        Choose between "cc" for connected components and one of "eom" or "flat" for 
        hierarchical single linkage.

    min_cluster_size: int, default=1
        Drop clusters smaller than the minimum cluster size. This parameter is passed to the
        hierarchical single linkage methods, or applied after the average threshold.

    Attributes
    ----------

    labels_ : array-like of shape (n_samples,)
        An array of labels for the data samples; this is a integer array as per other scikit-learn
        clustering algorithms. A value of -1 indicates that a point is a noise point and
        not in any cluster.

    cohesion_ : array-like of shape (n_samples, n_samples)
        A matrix of the cohesion value C[x, w]. This matrix is not symmetric.
        Can be thought of a complete weighted directed graph on n_samples vertices.
    """

    def __init__(self, n_neighbors=100, metric="euclidean", cluster_selection_method="eom", threshold=None, min_cluster_size=1):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method
        self.threshold = threshold
        self.min_cluster_size = min_cluster_size

    def fit(self, X, y=None):
        """
        Fit the model to the data

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            The data to cluster.

        y : array-like of shape (n_samples,), default=None
            Ignored. This parameter exists only for compatibility with
            scikit-learn's fit_predict method.

        Returns
        -------

        self : PAKNNLD
               For compatibility with sklearn api.
        """
        X = check_array(X)
        n_neighbors = self.n_neighbors
        if n_neighbors > X.shape[0]:
            warnings.warn(
                "Asked for more neighbors that data points, defaulting to n_neighbors = n points. Consider using the PALD object for exact computations."
            )
            n_neighbors = X.shape[0]

        query_n_neighbors = min(5, self.n_neighbors//5)
        index = pynndescent.NNDescent(X, n_neighbors=query_n_neighbors, metric=self.metric)
        knn, knn_dist = index.query(X, k=self.n_neighbors)

        other_knn_uxy_sizes = make_other_knn_uxy_sizes(knn)
        n_closer_than = make_n_closer_than(knn, knn_dist)
        cohesion = n_closer_than / other_knn_uxy_sizes

        self.cohesion_ = sp.lil_array((X.shape[0], X.shape[0]), dtype=cohesion.dtype)
        for i in range(knn.shape[0]):
            self.cohesion_[i, knn[i, :]] = cohesion[i, :]
        self.cohesion_ = self.cohesion_.tocsr()
        symmetric_cohesion = self.cohesion_.minimum(self.cohesion_.transpose())

        if self.threshold is not None:
            threshold = np.quantile(symmetric_cohesion.data, self.threshold)
        else:
            threshold = 0
        
        if threshold > 0:
            symmetric_cohesion.data[symmetric_cohesion.data < threshold] = 0
            symmetric_cohesion.eliminate_zeros()
        symmetric_cohesion.setdiag(0)
        symmetric_cohesion.eliminate_zeros()

        if len(symmetric_cohesion.data) == 0: # Matrix may be empty
            self.labels_ = np.arange(symmetric_cohesion.shape[0])
        elif self.cluster_selection_method == "cc":
            self.labels_ = sp.csgraph.connected_components(symmetric_cohesion)[1]
        elif self.cluster_selection_method in ["eom", "leaf"]:
            self.labels_, self.condensed_tree_ = hslc(self.cohesion_, self.cluster_selection_method, self.min_cluster_size)
        else:
            raise ValueError(f"cluster_selection_method should be one of 'cc', 'eom', or 'leaf'. Got {self.cluster_selection_method}")

        if self.min_cluster_size > 1 and self.cluster_selection_method not in ['eom', 'leaf']:
            prune_clusters(self.labels_, self.min_cluster_size)

        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True

        return self
