import pytest
import numpy as np
from sklearn.utils.estimator_checks import check_estimator

from tripletclustering import PALD


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


def test_pald_internals(data):
    pald = PALD()
    pald.fit(data)
    assert hasattr(pald, "is_fitted_")
    np.testing.assert_equal(pald.labels_, [0, 0, 0, 0, 1, 1, 1, 2])


def test_pald_eom(data):
    pald = PALD(cluster_selection_method="eom", min_cluster_size=2)
    pald.fit(data)
    assert hasattr(pald, "is_fitted_")
    np.testing.assert_equal(pald.labels_, [0, 0, 0, 0, 1, 1, 1, -1])


def test_pald_is_sklearn_estimator():
    pald = PALD()
    check_estimator(pald)
