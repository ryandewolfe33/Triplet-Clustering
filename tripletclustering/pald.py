import numpy as np
import scipy.sparse as sp
from numba import njit, prange
from numba_progress import ProgressBar
from sklearn.metrics import pairwise_distances
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array

from .clustering_utils import hslc, prune_clusters


@njit(parallel=True, nogil=True, fastmath=True)
def cohesion(D, progress_bar):
    """
    Compute the C[x, w] cohesion matrix. Algorithm taken from
    https://www.pnas.org/doi/suppl/10.1073/pnas.2003634119#supplementary-materials
    """
    n = D.shape[0]
    C = np.zeros_like(D, dtype="float32")
    for i in prange(n):
        for j in prange(i):
            wi = 0
            n_uij = 0
            for k in range(n):
                if D[i,k] <= D[i,j] or D[j,k] <= D[i,j]:
                    n_uij += 1
            #assert n_uij > 0
            for k in range(n):
                if D[i,k] <= D[i,j] or D[j,k] <= D[i,j]:
                    if D[i,k] < D[j,k]:
                        C[i,k] += 1 / n_uij
                    elif D[j,k] < D[i,k]:
                        C[j,k] += 1 / n_uij
                    else:
                        C[i,k] += 0.5 / n_uij
                        C[j,k] += 0.5 / n_uij
            progress_bar.update()
    C = C / (n - 1)
    return C


class PALD(ClusterMixin, BaseEstimator):
    """
    Partitioned Local Depth clustering algorithm. Based on comparisons between distances,
    so scales decently to high dimensions. O(n^3) run time and O(n^2) space requirements so
    not recommended for more than a few thousand points. Based on the paper "A social perspective
    on perceived distances reveals deep community structure" by Kenneth Berenhaut, Katherine
    Moorea, and Ryan Melvin.
    https://www.pnas.org/doi/10.1073/pnas.2003634119

    Parameters
    ----------

    metric : string, default="euclidean"
        Passed to sklearn.metrics.pairwise_distances so a wide range of options are supported.
    
    cluster_selection_method: string, default="strong"
        Choose between particularly strong cohesion (like in the PALD paper), a user specified
        threshold, and hierarchical single linkage based methods of 'eom' and 'leaf'.

    threshold: float, optional
        Single linkage cluster the cohesion graph at this cut level. Only used if
        cluster_selection_method is 'threshold'

    min_cluster_size: int, default=1
        Drop clusters smaller than the minimum cluster size. This parameter is passed to the
        hierarchical single linkage methods, or applied after the average threshold.

    verbose: bool, default=True
        Flag to print progress information

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

    def __init__(self, metric="euclidean", cluster_selection_method="strong", threshold=None, min_cluster_size=1, verbose=False):
        self.metric = metric
        self.threshold = threshold
        self.cluster_selection_method = cluster_selection_method
        self.min_cluster_size = min_cluster_size
        self.verbose=verbose

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

        self : PALD
               For compatibility with sklearn api.

        """
        if self.metric == "precomputed":
            X = check_array(X, ensure_all_finite=False)
            D = X
        else:
            X = check_array(X)
            print("Computing pairwise distance matrix") if self.verbose else None
            D = pairwise_distances(X, metric=self.metric)
        print("Computing cohesion matrix") if self.verbose else None
        with ProgressBar(
            total = D.shape[0]*(D.shape[0]-1)//2,
            disable = not self.verbose,
        ) as progress_bar:
            self.cohesion_ = cohesion(D, progress_bar)
        print("Clustering") if self.verbose else None
        symmetric_cohesion = np.minimum(self.cohesion_, self.cohesion_.T)
        if self.cluster_selection_method in ["strong", "threshold"]:
            self.threshold_ = np.mean(np.diagonal(self.cohesion_)) / 2 if self.cluster_selection_method == "strong" else self.threshold
            symmetric_cohesion[symmetric_cohesion < self.threshold_] = 0
            self.labels_ = sp.csgraph.connected_components(symmetric_cohesion)[1]
        elif self.cluster_selection_method == "threshold":
            if self.threshold is None:
                raise ValueError("Must set a threshold value if using cluster_selection_method is threshold.")
            self.labels_ = threshold_clusters(self.cohesion)
        elif self.cluster_selection_method in ["eom", "leaf"]:
            self.labels_, self.condensed_tree_ = hslc(self.cohesion_, self.cluster_selection_method, self.min_cluster_size)
        else:
            raise ValueError(f"cluster_selection_method should be one of 'average', 'eom', or 'leaf'. Got {self.cluster_selection_method}")

        if self.min_cluster_size > 1 and self.cluster_selection_method not in ["eom", "leaf"]:
            prune_clusters(self.labels_, self.min_cluster_size)

        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True

        return self
