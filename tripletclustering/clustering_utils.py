import numpy as np
import scipy.sparse as sp
from numba import njit
import fast_hdbscan


def hslc(similarity_graph, cluster_selection_method, min_cluster_size):
    if isinstance(similarity_graph, np.ndarray):
        distance_graph = 1 / similarity_graph
    elif sp.issparse(similarity_graph):
        distance_graph = similarity_graph.copy()
        distance_graph.data = 1 / distance_graph.data
    else:
        raise ValueError(
            "similarity_graph must be either a numpy array or a scipy.sparse array."
        )
    mst = sp.csgraph.minimum_spanning_tree(distance_graph)
    # If necessary, add max_dist edges to make it connected
    n_cc, cc_labels = sp.csgraph.connected_components(mst, directed=False)
    if n_cc > 1:
        max_dist = np.max(mst.data)
        mst = mst.todok()
        for i in range(np.max(cc_labels) + 1):
            first_id_in_cc = np.min(np.argwhere(cc_labels == i).reshape(-1))
            # Connect to vertex 0 with edge weight 0.1
            if first_id_in_cc != 0:
                mst[0, first_id_in_cc] = max_dist
    # Format MST for sinlge_linkage
    mst = mst.tocoo()
    mst = np.vstack((mst.row, mst.col, mst.data)).transpose()
    clusters, membership_strengths, linkage_tree, condensed_tree, sorted_mst = (
        fast_hdbscan.hdbscan.clusters_from_spanning_tree(
            mst,
            cluster_selection_method=cluster_selection_method,
            min_cluster_size=min_cluster_size,
        )
    )
    return clusters, condensed_tree


@njit
def prune_clusters(labels, min_cluster_size):
    "Drop labels with fewer than min_cluster_size objects and relabel to 0-N"
    labels, label_sizes = np.unique(labels, return_counts=True)
    keep_labels = labels[label_sizes >= min_cluster_size]
    new_id_dict = {keep_labels[i]: i for i in range(len(keep_labels))}
    for i in range(len(labels)):
        if labels[i] not in new_id_dict:
            labels[i] == -1
        else:
            labels[i] = new_id_dict[labels[i]]
