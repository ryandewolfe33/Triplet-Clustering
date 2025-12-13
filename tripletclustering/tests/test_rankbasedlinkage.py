import pytest
import numpy as np
import scipy.sparse as sp
from sklearn.utils.estimator_checks import check_estimator

from tripletclustering import RankBasedLinkage


@pytest.fixture
def data():
    # Data is copied from exdata1 in R package
    X = np.array(
        [
            [0, 7, 6, 4, 8, 3, 1, 5, 9, 2],
            [5, 0, 3, 6, 7, 8, 1, 4, 9, 2],
            [7, 6, 0, 1, 9, 5, 3, 4, 8, 2],
            [2, 9, 5, 0, 4, 6, 3, 8, 7, 1],
            [9, 8, 7, 4, 0, 3, 6, 2, 1, 5],
            [4, 9, 3, 6, 5, 0, 1, 7, 8, 2],
            [1, 6, 5, 4, 8, 3, 0, 7, 9, 2],
            [7, 9, 1, 8, 2, 3, 5, 0, 4, 6],
            [9, 6, 7, 5, 1, 3, 8, 2, 0, 4],
            [7, 5, 4, 2, 8, 3, 1, 6, 9, 2],
        ]
    )
    yield X


@pytest.fixture
def vectors():
    rng = np.random.default_rng(seed=42)
    X = rng.uniform(-1, 1, (25, 5))
    yield X


def test_rankbasedlinkage_class(data):
    rbl = RankBasedLinkage(
        n_neighbors=10, metric="precomputed", cluster_selection_method="threshold"
    )
    predict = rbl.fit(data)

    answer = np.array(
        [
            [0, 2, 2, 5, 0, 5, 8, 1, 0, 2],
            [0, 0, 3, 0, 0, 0, 3, 0, 0, 4],
            [0, 0, 0, 4, 0, 4, 4, 5, 0, 5],
            [0, 0, 0, 0, 2, 3, 5, 1, 1, 7],
            [0, 0, 0, 0, 0, 2, 0, 6, 8, 0],
            [0, 0, 0, 0, 0, 0, 6, 2, 1, 6],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 7],
            [0, 0, 0, 0, 0, 0, 0, 0, 5, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    dense = rbl.linkage_.todense()
    assert np.array_equal(answer, dense)

    labels = np.array([0, 1, 2, 0, 3, 0, 0, 3, 3, 0])
    assert np.array_equal(labels, rbl.labels_)


def test_rankbasedlinkage_eom(data):
    rbl = RankBasedLinkage(
        n_neighbors=10,
        metric="precomputed",
        cluster_selection_method="eom",
        min_cluster_size=2,
    )
    predict = rbl.fit(data)
    labels = np.array([1, -1, 0, 1, 0, 1, 1, 0, 0, 1])
    assert np.array_equal(labels, rbl.labels_)


def test_rankbasedlinkage_leaf(data):
    rbl = RankBasedLinkage(
        n_neighbors=10,
        metric="precomputed",
        cluster_selection_method="leaf",
        min_cluster_size=2,
    )
    predict = rbl.fit(data)
    labels = np.array([2, -1, 0, 1, 0, -1, 2, 0, 0, 1])
    assert np.array_equal(labels, rbl.labels_)


def test_rankbasedlinkage_knn(vectors):
    rbl = RankBasedLinkage(n_neighbors=10, metric="cosine", min_cluster_size=15)
    predict = rbl.fit(vectors)
    labels = np.full(25, -1)
    assert np.array_equal(labels, rbl.labels_)


def test_rankbasedlinkage_is_sklearn_estimator():
    rbl = RankBasedLinkage(metric="euclidean")
    check_estimator(rbl)
