from numba import njit
import numpy as np
import scipy.sparse as sp

@njit
def extract_flat_from_condensed_tree(
    condensed_tree, # Works for either condensed or cluster tree
    cut_value,
):
    # Sort by lambda
    lambda_sort = np.argsort(condensed_tree.lambda_val)
    parent = condensed_tree.parent[lambda_sort]
    child = condensed_tree.child[lambda_sort]
    lambda_val = condensed_tree.lambda_val[lambda_sort]
    child_size = condensed_tree.child_size[lambda_sort]

    # only way to create a typed empty set
    selected = {np.int64(0)}
    selected.clear()
    # For each cluster, keep if lambda > threshold and parent not yet kept.
    for i in range(len(parent)):
        if (
            lambda_val[i] >= cut_value
            and parent[i] not in selected
            and child_size[i] > 1
        ):
            selected.add(child[i])
    selected_array = np.empty(len(selected), dtype="int64")
    i = 0
    for j in selected:
        selected_array[i] = j
        i += 1
    return selected_array


def compute_pald_cut_value(similarity):
    cut_value = np.mean(similarity.diagonal()) / 2
    return cut_value


def compute_rbl_cut_value(similarity):
    n = similarity.shape[0]
    triu = sp.triu(similarity)
    triu.eliminate_zeros()
    edge_weights = triu.data
    edge_count, cut_values = np.histogram(edge_weights, bins=np.ptp(edge_weights))
    kept_edges = np.cumsum(edge_count)
    cut_value = cut_values[np.searchsorted(kept_edges, n, side="right")]  
    return cut_value

def flat(
    condensed_tree,
    similarity,
    cut_value
):
    if cut_value == "pald":
        cut_value = compute_pald_cut_value(similarity)
    elif cut_value == "rbl":
        cut_value = compute_rbl_cut_value(similarity)  
    elif not np.issubdtype(type(cut_value), float):
        raise ValueError("Unknown cut_value type.")
    selected_clusters = extract_flat_from_condensed_tree(condensed_tree, cut_value)
    return selected_clusters
