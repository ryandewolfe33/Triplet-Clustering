import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import (
    check_array,
    check_random_state,
)
from sklearn.utils.validation import validate_data
from fast_hdbscan.cluster_trees import (
    mst_to_linkage_tree,
    condense_tree,
    get_cluster_label_vector,
)
from fast_hdbscan.precomputed import bridge_forest_with_inf

from .similarity.ood import knn_rank_graph, full_ood


def build_mst(similarity: sp.sparray | np.ndarray) -> np.ndarray:
    similarity = check_array(
        similarity, accept_sparse=True, ensure_min_samples=0, ensure_min_features=0
    )
    if similarity.shape == (0, 0):
        return np.empty((0, 3))
    similarity = sp.csr_array(similarity)
    similarity.eliminate_zeros()
    similarity.data = 1 / similarity.data
    mst = sp.csgraph.minimum_spanning_tree(similarity)
    similarity.data = 1 / similarity.data  # Undo inplace 1/data
    n_cc, cc_labels = sp.csgraph.connected_components(mst, directed=False)
    mst = mst.tocoo()
    mst = np.array([mst.row, mst.col, mst.data])
    sort_order = np.argsort(mst[2])
    mst = mst[:, sort_order]
    mst = mst.T
    if n_cc > 1:
        mst = bridge_forest_with_inf(mst, cc_labels, similarity.shape[0])
    return mst


class TripletClustering(ClusterMixin, BaseEstimator):
    """ Clustering algorithm based on triplet comparisons.

    Parameters:
    -----------

    similarity_method: string (optional, default "rbl")
        The method used to measure similarity between points. Options are "rbl" for rank-based-linkage,
        "pald" for partitoned-local-depth, or "paknnld" for partitioned-k-nearest-neighbor-local-depth.
    
    cluster_selection_method: string (optional, default "eom)
        The method used to select clusters from the condensed single linkage dendrogram. Options are
        "eom" for excess of mass, "leaf", "pl" for persistent leaves, or "flat" for a flat horizontal cut.

    min_cluster_size: int (optional, default 15)
        Minimum cluster size used to prune the single linkage dendrogram.
    
    cut_value: str | float (optional, default "method")
        Method or value to cut the condensed dendrogram at; only used when cluster_selection_method="flat".
        Options are "pald", "rbl", "method" (used to pass the similarity_method), or a positive value.
    
    metric: str (optional, default "cosine")
        Metric to determine the distance between points. Will be passed to pynndescent or sklearn's pairwise distances.

    n_neighbors: int (optional, default 15)
        The number of nearest neighbors to rank in the out-ordered digraph for the "rbl" and "paknnld" similarity methods.

    knn_kwargs: dict (optional)
        Key word arguments that are passed to pynndescent. Must not contain random_state, n_neighbors, or metric.
    
    cluster_selection_kwargs: dict (optional)
        Key word arguments that are passed to the cluster selection method.
    
    save_steps: tuple (optional)
        Flag to save certain intermediate steps for later processing or visualization. Can include
        ("similarity", "linkage_tree", "condensed_tree").
    
    random_state: int | np.random.RandomState | np.random.Generator (optional)
        Random state; only used in pynndescent.
    
    verbose: bool (optional, default False)
        Flag to print progress messages and progress bars.
    """

    def __init__(
        self,
        similarity_method: str = "rbl",
        cluster_selection_method: str = "eom",
        min_cluster_size: int = 15,
        cut_value: str | float = "method",
        metric: str = "cosine",
        n_neighbors: int = 50,
        knn_kwargs: dict | None = None,
        cluster_selection_kwargs: dict | None = None,
        save_steps: tuple | None = None,
        random_state: int | np.random.RandomState | np.random.Generator | None = None,
        verbose: bool = False,
    ):
        self.similarity_method = similarity_method
        self.cluster_selection_method = cluster_selection_method
        self.min_cluster_size = min_cluster_size
        self.cut_value = cut_value
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.knn_kwargs = knn_kwargs
        self.cluster_selection_kwargs = cluster_selection_kwargs
        self.save_steps = save_steps
        self.random_state = random_state
        self.verbose = verbose

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags

    def _validate_params(self):
        available_similarity_methods = ["pald", "paknnld", "rbl", "precomputed"]
        if self.similarity_method not in available_similarity_methods:
            raise ValueError(
                f"similarity_method must be one of: {available_similarity_methods}. "
            )

        available_cluster_selection_methods = ["eom", "leaf", "pl", "flat"]
        if self.cluster_selection_method not in available_cluster_selection_methods:
            raise ValueError(
                f"similarity_method must be one of: {available_cluster_selection_methods}. "
            )

        if (
            not np.issubdtype(type(self.min_cluster_size), int)
            or self.min_cluster_size <= 0
        ):
            raise ValueError("min_cluster_size must be a positive integer.")

        if (
            self.cluster_selection_method == "flat"
            and self.cut_value != "method"
            and (not np.issubdtype(type(self.cut_value), float) or self.cut_value <= 0)
        ):
            raise ValueError("cut_value must be 'method' or a positive float.")

        if not np.issubdtype(type(self.n_neighbors), int) or self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be a positive integer.")

    def fit(self, X, y=None):
        self._validate_params()
        X = validate_data(self, X, accept_sparse=True, ensure_all_finite=False)
        n = X.shape[0]
        (
            print(
                f"Running Triplet Clustering with similarity {self.similarity_method}, min cluster size {self.min_cluster_size}, and cluster selection method {self.cluster_selection_method}."
            )
            if self.verbose
            else None
        )
        print("Building Similarity Matrix") if self.verbose else None
        knn_kwargs = self.knn_kwargs if self.knn_kwargs is not None else {}
        random_state = check_random_state(self.random_state)
        if self.similarity_method == "precomputed":
            similarity = sp.csr_array(X)
            validate_precomputed_sparse_graph(similarity)
        else:
            if self.similarity_method == "pald":
                from .similarity.pald import pald

                ood = full_ood(X, self.metric)
                similarity = pald(ood, verbose=self.verbose)
            elif self.similarity_method == "paknnld":
                from .similarity.paknnld import paknnld

                ood = knn_rank_graph(
                    X,
                    n_neighbors=self.n_neighbors,
                    metric=self.metric,
                    random_state=random_state,
                    **knn_kwargs,
                )
                similarity = paknnld(ood, verbose=self.verbose)
            elif self.similarity_method == "rbl":
                from .similarity.rbl import rbl

                ood = knn_rank_graph(
                    X,
                    n_neighbors=self.n_neighbors,
                    metric=self.metric,
                    random_state=random_state,
                    **knn_kwargs,
                )
                similarity = rbl(ood, verbose=self.verbose)

        print("Building Condensed Tree") if self.verbose else None
        sorted_mst = build_mst(similarity)
        linkage_tree = mst_to_linkage_tree(sorted_mst)
        condensed_tree = condense_tree(
            linkage_tree, min_cluster_size=self.min_cluster_size
        )
        print("Selecting Clusters") if self.verbose else None
        cluster_selection_kwargs = (
            self.cluster_selection_kwargs
            if self.cluster_selection_kwargs is not None
            else {}
        )
        if self.cluster_selection_method == "leaf":
            from .cluster_selection.eom_leaf import eom_leaf

            selected_clusters = eom_leaf(
                condensed_tree,
                n,
                cluster_selection_method="leaf",
                **cluster_selection_kwargs,
            )
        elif self.cluster_selection_method == "eom":
            from .cluster_selection.eom_leaf import eom_leaf

            selected_clusters = eom_leaf(
                condensed_tree,
                n,
                cluster_selection_method="eom",
                **cluster_selection_kwargs,
            )
        elif self.cluster_selection_method == "pl":
            from .cluster_selection.pl import pl

            self.labels_, self.extras = pl(
                condensed_tree, n, self.min_cluster_size, n, **cluster_selection_kwargs
            )
            selected_clusters = None
        elif self.cluster_selection_method == "flat":
            from .cluster_selection.flat import flat

            if self.cut_value == "method":
                selected_clusters = flat(
                    condensed_tree, similarity, self.similarity_method
                )
            else:
                selected_clusters = flat(condensed_tree, similarity, self.cut_value)
        else:
            raise ValueError(
                f"cluster_selection_method '{self.cluster_selection_method}' not recognised."
            )

        if selected_clusters is not None:
            self.labels_ = get_cluster_label_vector(
                condensed_tree, selected_clusters, 0.0, n
            )

        if self.save_steps is not None:
            if "similarity" in self.save_steps:
                self.similarity_ = similarity
            if "linkage_tree" in self.save_steps:
                self.linkage_tree_ = linkage_tree
            if "condensed_tree" in self.save_steps:
                self.condensed_tree_ = condensed_tree

        print("Done Triplet Clustering") if self.verbose else None
        return self

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_
