# Copyright (c) 2023, Tutte Institute for Mathematics and Computing
# Copyright (c) 2026 Ryan DeWolfe

import numpy as np
import scipy as sp
from typing import Optional
from fast_hdbscan.cluster_trees import (
    extract_leaves,
    get_cluster_label_vector,
    get_point_membership_strength_vector,
    mask_condensed_tree,
)
from fast_hdbscan.layer_clusters import (
    min_cluster_size_barcode,
    compute_total_persistence,
    find_peaks,
    select_diverse_peaks,
    extract_clusters_by_id,
)


def pl(
    condensed_tree,
    n_samples,
    min_cluster_size,
    layer_similarity_threshold=0.2,
    max_layers=10,
    verbose=False,
):
    cluster_layers = []
    persistence_scores = []
    sizes = []
    total_persistence = 0.0

    print("Finding optimal resolution layers ...") if verbose else None

    mask = condensed_tree.child >= n_samples
    cluster_tree = mask_condensed_tree(condensed_tree, mask)

    # Check if cluster_tree is valid before processing
    if len(cluster_tree.child) > 0 and cluster_tree.child[-1] >= n_samples:
        leaves = extract_leaves(condensed_tree)
        clusters = get_cluster_label_vector(condensed_tree, leaves, 0.0, n_samples)

        births, deaths, parents, lambda_deaths = min_cluster_size_barcode(
            cluster_tree, n_samples, min_cluster_size
        )
        sizes, total_persistence = compute_total_persistence(
            births, deaths, lambda_deaths
        )
        peaks = find_peaks(total_persistence)
    else:
        # Handle empty or invalid cluster tree
        clusters = np.full(n_samples, -1)
        births = np.array([])
        deaths = np.array([])
        parents = np.array([])
        lambda_deaths = np.array([])
        sizes = np.array([])
        total_persistence = np.array([])
        peaks = np.array([], dtype=np.int64)

    # Always include the base layer (from initial condensed tree)
    cluster_layers.append(clusters)
    persistence_scores.append(0.0)  # Base layer gets 0 persistence score

    # Select diverse peaks using hierarchical selection
    selected_peaks = select_diverse_peaks(
        peaks,
        total_persistence,
        sizes,
        births,
        deaths,
        min_similarity_threshold=layer_similarity_threshold,
        max_layers=max_layers - 1,  # Reserve one slot for base layer
    )

    for peak in selected_peaks:
        best_birth = sizes[peak]
        persistence = total_persistence[peak]
        selected_clusters = (
            np.where((births <= best_birth) & (deaths > best_birth))[0] + n_samples
        )
        labels, strengths = extract_clusters_by_id(condensed_tree, selected_clusters)
        cluster_layers.append(labels)
        persistence_scores.append(persistence)

    # Sort cluster layers by number of clusters (most clusters first)
    n_clusters_per_layer = [layer.max() + 1 for layer in cluster_layers]
    sorted_indices = np.argsort(n_clusters_per_layer)[::-1]  # Descending order

    cluster_layers = [cluster_layers[i] for i in sorted_indices]
    persistence_scores = [persistence_scores[i] for i in sorted_indices]

    best_layer = np.argmax(persistence_scores)
    labels = cluster_layers[best_layer]

    return (
        labels,
        {
            "cluster_layers": cluster_layers,
            "persistence_scores": persistence_scores,
            "sizes": sizes,
        },
    )
