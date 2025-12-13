import scipy.sparse as sp
import numpy as np
import pynndescent
from numba import njit, prange
from numba.typed import List
from numba.types import UniTuple, int32
from numba_progress import ProgressBar
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array

from .clustering_utils import hslc, prune_clusters


def knn_rank_graph(X, n_neighbors, metric):
    query_n_neighbors = min(5, n_neighbors//5)
    index = pynndescent.NNDescent(X, n_neighbors=query_n_neighbors, metric=metric)
    # Get one more neighbor than n_neighbors so we can drop the point itself
    neighbors = index.query(X, k=n_neighbors+1)[0]
    neighbors = neighbors[:, 1:] # Drop the point itself
    ood = sp.lil_array((neighbors.shape[0], neighbors.shape[0]), dtype="int32")
    for i in range(neighbors.shape[0]):
        ood[i, neighbors[i, :]] = np.arange(n_neighbors, dtype="int32") + 1
    ood = ood.tocsr()
    return ood


#TODO numbaify
def precomputed_to_ood(X, n_neighbors):
    X = sp.csr_array(X)
    for row in range(X.shape[0]):
        row_neighbors = X.indices[X.indptr[row]:X.indptr[row+1]]
        row_data = X.data[X.indptr[row]:X.indptr[row+1]]
        if len(row_neighbors) <= n_neighbors:
            continue
        argsort = np.argsort(row_data)
        if row_neighbors[argsort[0]] == row:
            offset = X.indptr[row]
            remove_indices = argsort[n_neighbors+1:] + offset
            X.data[remove_indices] = 0
            X.data[argsort[0] + offset] = 0
        else:
            offset = X.indptr[row]
            remove_indices = argsort[n_neighbors:] + offset
            X.data[remove_indices] = 0
    X.eliminate_zeros()
    return X


@njit
def mutual_friend_list(indptr, indices, tuple_type, progress_bar):
    rows = List([set(indices[indptr[i]:indptr[i+1]]) for i in range(len(indptr)-1)])
    L = List.empty_list(tuple_type)
    for x, x_friends in enumerate(rows):
        if len(x_friends) == 0:
            continue
        for z in x_friends:
            if x in rows[z] and x < z:
                L.append((int32(x), z))
        #progress_bar.update()
    return L


@njit(nogil=True)
def csc_lookup(indptr, indices, data, i, j):
    col = indices[indptr[j]:indptr[j+1]]
    offset = np.searchsorted(col, i)
    if col[offset] == i:
        return data[indptr[j]+offset]
    return 0


@njit
def in_sway(ood_indptr, ood_indices, ood_data, L, progress_bar):
    # Linkage graph is same vertices, non-negative integer edge weights
    row_ind = np.empty(len(L), dtype="int32")
    col_ind = np.empty(len(L), dtype="int32")
    data = np.empty(len(L), dtype="int32")

    for i in range(len(L)):
        x,z = L[i]
        y_ranks_x = set(ood_indices[ood_indptr[x]:ood_indptr[x+1]])
        y_ranks_z = set(ood_indices[ood_indptr[z]:ood_indptr[z+1]])
        ranks_either = len(y_ranks_x.union(y_ranks_z))

        xz = csc_lookup(ood_indptr, ood_indices, ood_data, x, z)
        for y in y_ranks_x:
            xy = csc_lookup(ood_indptr, ood_indices, ood_data, x, y)
            if xz < xy: # Remove if xy does not beat xz
                y_ranks_x.remove(y)

        zx = csc_lookup(ood_indptr, ood_indices, ood_data, z, x)
        for y in y_ranks_z:
            zy = csc_lookup(ood_indptr, ood_indices, ood_data, z, y)
            if zx < zy: # Remove if zy does not beat zx
                y_ranks_z.remove(y)

        in_sway = ranks_either - len(y_ranks_x.union(y_ranks_z))
        row_ind[i] = x
        col_ind[i] = z
        data[i] = in_sway
        progress_bar.update()

    return row_ind, col_ind, data
    

class RankBasedLinkage(ClusterMixin, BaseEstimator):
    """
    An unsupervised clustering algorithm for comparator data. Data can be passed as distances or
    vectors (and neighbors will be ranked by distance).  

    Parameters
    ----------

    n_neighbors: int, default=15
        The maximum number of rankings for each object.

    metric : string, default="precomputed"
        Distance to used to compare data. If precomputed, input distances will be used as the ranking.
    
    cluster_selection_method: string, default="eom"
        The method for determining the clusters from the in-sway graph. Choose between "eom" or "leaf" for
        hierarchical single linkage methods or "threshold" for a single linkage threshold.

    threshold: float, optional
        Single linkage cluster the cohesion graph at this cut level. Only used if cluster_selection_method
        is set to "threshold". If None uses the suggest threshold from the paper.

    min_cluster_size: int, default=1
        Drop clusters smaller than the minimum cluster size. This parameter is passed to the
        hierarchical single linkage methods, or applied after the threshold.

    verbose: bool, default=False
        Flag to print progress information

    Attributes
    ----------
    
    labels_ : array-like of shape (n_samples,)
        An array of labels for the data samples; this is a integer array as per other scikit-learn
        clustering algorithms. A value of -1 indicates that a point is a noise point and
        not in any cluster.

    insway_ : csr_array of shape (n_samples, n_samples)
        Adjacency matrix for the insway graph.
        The graph is weighted and directed on n_samples vertices.
    """

    def __init__(self, n_neighbors=15, metric="precomputed", cluster_selection_method="eom", threshold=None, min_cluster_size=1, verbose=False):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.n_neighbors=n_neighbors
        self.cluster_selection_method = cluster_selection_method
        self.threshold=threshold
        self.min_cluster_size = min_cluster_size
        self.verbose = verbose

    def fit(self, X, y=None):
        """
        Computes the insway matrix and cluster labels.
        """
        if self.metric == "precomputed":
            X = check_array(X, ensure_all_finite=False)
            self.ood_ = precomputed_to_ood(X, self.n_neighbors)
        else:
            X = check_array(X)
            self.ood_ = knn_rank_graph(X, self.n_neighbors, self.metric)
        
        print("Making Mutual Friend List") if self.verbose else None
        with ProgressBar(
            total = self.ood_.shape[0],
            disable = not self.verbose,
        ) as progress_bar:
            L = mutual_friend_list(self.ood_.indptr, self.ood_.indices, UniTuple(int32, 2), progress_bar)
        
        print("Compute In Sway") if self.verbose else None
        with ProgressBar(
            total = len(L),
            disable = not self.verbose,
        ) as progress_bar:
            ood = self.ood_.tocsc()
            row_ind, col_ind, data = in_sway(ood.indptr, ood.indices, ood.data, L, progress_bar)
        self.linkage_ = sp.coo_array((data, (row_ind, col_ind)), shape=(X.shape[0], X.shape[0]))
        
        print("Compute Clusters") if self.verbose else None
        if self.cluster_selection_method == "threshold":
            self.threshold_ = self.threshold
            if self.threshold_ is None:
                self.threshold_ = 0
                while np.sum(self.linkage_.data >= self.threshold_) >= self.linkage_.shape[0]:
                    self.threshold_ += 1
            pruned = self.linkage_.copy()
            pruned.data[pruned.data < self.threshold_] = 0
            pruned.eliminate_zeros()
            self.labels_ = sp.csgraph.connected_components(pruned)[1]
        elif self.cluster_selection_method in ["eom", "leaf"]:
            self.labels_, self.condensed_tree_ = hslc(self.linkage_, self.cluster_selection_method, self.min_cluster_size)
        else:
            raise ValueError(f"cluster_selection_method must be one of 'threshold', 'eom', or 'leaf'. Got {self.cluster_selection_method}")
        
        if self.min_cluster_size > 1 and self.cluster_selection_method not in ["eom", "leaf"]:
            prune_clusters(self.labels_, self.min_cluster_size)
        
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True

        return self

            