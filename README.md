# Triplet Clustering
A collection of clustering algorithms based on triplet comparisons (x is more similar to y than to x).

# Currently under renovation

This is a pure python implementation of two clustering algorithms based off the paper *A social perspective on perceived distances reveals deep community structure*. 

The PALD algorithm is directly from the paper. The official implemnetation is an R package available at https://github.com/moorekatherine/pald. It is a comparison based approach so scales well to higher dimensions. However it has a runtime of O(n^3) and a space complexity of O(n^2) so is only useful up to
thousands of points.

The PAKNNLD algorithm is an attempt to efficiently capture the PALD idea. Using only the K nearest neighbors, the runtime is probably O(n * log n) and can be used on hundreds of thousands of vectors.
The idea was mentioned in  *Partitioned K-nearest neighbor local depth for scalable comparison-based learning* but not thoroughly explored. A high level overview of the algorithm can be found in [this notebook](/notebooks/how_paknnld_works.ipynb)

Both algorithms are implemented according to the sklearn api so can be used in conjunction with other sklearn classes.

## Getting Started
For now the package must be cloned and locally installed.
```bash
git clone git@github.com:ryandewolfe33/PALD.py.git
cd PALD.py
pip install .
```

## Example Usage
```python
import sklearn.datasets as data
import pald
moons, _ = data.make_moons(n_samples=50, noise=0.05, random_state=123)
pald_labels = pald.PALD().fit_predict(data)
paknnld_labels = pald.PAKNNLD().fit_predict(data)
```

## Some dataset information
Test data was taken from https://github.com/moorekatherine/pald/tree/main/data and converted to csv files (after dropping headers) using https://github.com/vnmabus/rdata. 

Aggregation.txt data is a Benchmark data set Aggregation consisting of n = 788 points.
Link to data: http://cs.joensuu.fi/sipu/datasets/Aggregation.txt
A. Gionis, H. Mannila, and P. Tsaparas, Clustering aggregation. ACM Transactions on Knowledge Discovery from Data (TKDD), 2007. 1(1): p. 1-30.

HDBSCAN comparison clusterable data is taken from the HDBSCAN repository. https://github.com/scikit-learn-contrib/hdbscan

Several openml datasets are used for comparison with EVoC.
