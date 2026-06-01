import pytest
import numpy as np

from tripletclustering.similarity.rbl import rbl
from tripletclustering.cluster_selection.flat import compute_rbl_cut_value


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
            [7, 5, 4, 2, 8, 3, 1, 6, 9, 0],
        ]
    )
    yield X


def test_rankbasedlinkage(data):
    predict = rbl(data)

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
    dense = predict.toarray()
    assert np.array_equal(answer, dense)

    cut_value = compute_rbl_cut_value(predict)
    assert cut_value == 2
