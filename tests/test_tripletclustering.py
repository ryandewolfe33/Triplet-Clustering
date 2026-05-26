import pytest
import numpy as np
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted

from tripletclustering import TripletClustering

@pytest.fixture
def data():
    # Data is copied from exdata1 in R package
    X = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 1.28],
            [-1.0, -1.28],
            [0.0, 0.0],
            [1.4, -0.9],
            [1.4, 0.0],
            [1.4, 0.9],
            [0.5, 3.0],
        ]
    )
    yield X


def test_default(data):
    tpc = TripletClustering()
    tpc.fit(data)
    check_is_fitted(tpc)

def test_pald(data):
    tpc = TripletClustering(similarity_method="pald")
    assert tpc.similarity_method == "pald"
    tpc.fit(data)
    check_is_fitted(tpc)

def test_paknnld(data):
    tpc = TripletClustering(similarity_method="paknnld")
    assert tpc.similarity_method == "paknnld"
    tpc.fit(data)
    check_is_fitted(tpc)

def test_leaf(data):
    tpc = TripletClustering(cluster_selection_method="leaf")
    assert tpc.cluster_selection_method == "leaf"
    tpc.fit(data)
    check_is_fitted(tpc)

def test_pl(data):
    tpc = TripletClustering(cluster_selection_method="pl")
    assert tpc.cluster_selection_method == "pl"
    tpc.fit(data)
    check_is_fitted(tpc)

def test_flat(data):
    tpc = TripletClustering(similarity_method="rbl", cluster_selection_method="flat", cut_value="method")
    tpc.fit(data)
    check_is_fitted(tpc)

    tpc = TripletClustering(similarity_method="pald", cluster_selection_method="flat", cut_value="method")
    tpc.fit(data)
    check_is_fitted(tpc)

    tpc = TripletClustering(cluster_selection_method="flat", cut_value=0.5)
    tpc.fit(data)
    check_is_fitted(tpc)

def test_is_sklearn_estimator():
    tpc = TripletClustering()
    check_estimator(tpc)