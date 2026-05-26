from fast_hdbscan.cluster_trees import (
    extract_eom_clusters,
    cluster_tree_leaves,
    cluster_epsilon_search,
    cluster_tree_from_condensed_tree,
    extract_clusters_bcubed,
)
import numpy as np


def eom_leaf(
    condensed_tree,
    n,
    cluster_selection_method="eom",
    data_labels=None,
    semi_supervised=False,
    ss_algorithm="bc",
    max_cluster_size=np.inf,
    allow_single_cluster=False,
    cluster_selection_epsilon=0.0,
    ):
    cluster_tree = cluster_tree_from_condensed_tree(condensed_tree)
    if cluster_selection_method == "eom":
        if semi_supervised:
            # Assumes ss_algorithm is either 'bc' or 'bc_simple'
            selected_clusters = extract_clusters_bcubed(
                condensed_tree,
                cluster_tree,
                data_labels,
                allow_virtual_nodes=True if ss_algorithm == "bc" else False,
                allow_single_cluster=allow_single_cluster,
            )
        else:
            selected_clusters = extract_eom_clusters(
                condensed_tree,
                cluster_tree,
                max_cluster_size=max_cluster_size,
                allow_single_cluster=allow_single_cluster,
            )
    elif cluster_selection_method == "leaf":
        if cluster_tree.parent.shape[0] == 0:
            selected_clusters = np.empty(0, dtype=np.int64)
        else:
            selected_clusters = cluster_tree_leaves(cluster_tree, n)
    else:
        raise ValueError(f"Invalid cluster_selection_method {cluster_selection_method}")

    if len(selected_clusters) > 1 and cluster_selection_epsilon > 0.0:
        selected_clusters = cluster_epsilon_search(
            selected_clusters,
            cluster_tree,
            min_epsilon=cluster_selection_epsilon,
        )

    return selected_clusters
