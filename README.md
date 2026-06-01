# Triplet Clustering
A collection of clustering algorithms based on triplet comparisons (x is more similar to y than to z).

## Getting Started
For now the package must be cloned and locally installed.
```bash
git clone git@github.com:ryandewolfe33/tripletclustering.git
cd tripletclustering
pip install .
```

## Example Usage
```python
import sklearn.datasets as data
from tripletclustering import TripletClustering
moons, _ = data.make_moons(n_samples=50, noise=0.05, random_state=123)
clusterer = TripletClustering()
labels = clusterer.fit_predict(data)
```

# How it works

This package implements a two-step clustering algorithm: 1) compute a similarity score between points then 2) select clusters from the induced condensed single linkage dendrogram.

The similarity scores are based on triplet comparisons, information in the form of "point a is closer to point b than point c".

From these similarity scores, we construct a single linkage dendrogram (the clusters at a cut level are the connected components of the graph with similarity greater than the cut level). Next, we construct a condensed dendrogram by dropping branches smaller than some minimum size (set with minimum_clsuter_size). Finally, we select clusters from the condensed dendrogram via one of the methods below.

For more information on dendrograms and pruning, see the first two sections of the following paper:
> Leland McInnes and John Healy. Accelerated Hierarchical Density Based Clustering. In: 2017 IEEE International Conference on Data Mining Workshops (ICDMW) (2017) - https://doi.org/10.1109/ICDMW.2017.12

or the docs of hdbscan:
https://hdbscan.readthedocs.io/en/latest/
> Leland McInnes, John Healy and Steve Astels. hdbscan: Hierarchical density based clustering. The Journal of Open Source Software, Volume 2, Number 11 (2017)

## Similarity Measures

### PALD
The PALD algorithm is directly from the paper. The official implemnetation is an R package available at https://github.com/moorekatherine/pald. It is a comparison based approach so scales well to higher dimensions. However it has a runtime of O(n^3) and a space complexity of O(n^2) so is only useful up to thousands of points.
> Kenneth S. Berenhaut, Katherine E. Moore, and Ryan L. Melvin. A social perspective on perceived distances reveals deep community structure. Proceedings of the National Academy of Sciences, Volume 119, Issue 4, number e2003634119 (2022) - https://doi.org/10.1073/pnas.2003634119 

### PAKNNDL
The PAKNNLD algorithm is an attempt to efficiently capture the PALD idea. Using only the K-nearest-neighbors, the runtime is probably O(n * log n) and can be used on hundreds of thousands of vectors. The idea was mentioned in the following paper
> Jacob D. Baron, R. W. R. Darling, J. Laylon Davis, and R. Pettit. Partitioned K-nearest neighbor local depth for scalable comparison-based learning. arXiv Preprint (2021) - https://doi.org/10.48550/arXiv.2108.08864


### Ranked-Based-Linkage (rbl)
The Ranked-Based-Linkage similarity measure also only uses the K-nearest-neighbors of each point, and the runtime is also likely O(n * log n).

> R. W. R. Darling, Will Grilliette, and Adam Logan. Rank-based linkage I: triplet comparisons and oriented simplicial complexes. Compositionality, Volume 8 (2026) - https://doi.org/10.46298/compositionality-8-2


## Cluster Selection Methods

### Excess of Mass (eom)
Select non-overlapping clusters to maximize the area on the condensed dendrogram.
> R.J.G.B. Campello, D. Moulavi, and J. Sander. Density-Based Clustering Based on Hierarchical Density Estimates. In: Advances in Knowledge Discovery and Data Mining. PAKDD 2013. Volume 7819 of Lecture Notes in Computer Science() (2013) - https://doi-org.ezproxy.lib.torontomu.ca/10.1007/978-3-642-37456-2_14

### Leaf
Return the leaves of the condensed dendrogram.

### Persistent Leaves (pl)
> Daniël Bot, Leland McInnes, and Jan Aerts. Persistent Multiscale Density-based Clustering. arXiv Preprint (2026) - https://doi.org/10.48550/arXiv.2512.16558

### Flat
Return clusters corresponding to a flat cut across the condensed dendrogram. The similarity methods PALD and Rank-Based-Linkage come with a method to choose a cut-value, or a custom value may be passed.
