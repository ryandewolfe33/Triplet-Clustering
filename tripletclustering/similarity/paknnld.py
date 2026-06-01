import numpy as np
from numba import njit, prange
from numba_progress import ProgressBar
import scipy.sparse as sp


@njit(nogil=True, parallel=True)
def cohesion(ood_indptr, ood_indices, ood_data, progress_bar):
    cohesion = np.empty_like(ood_indices, dtype="float64")
    for x in prange(len(ood_indptr) - 1):
        x_friends = ood_indices[ood_indptr[x] : ood_indptr[x + 1]]
        x_ranks = ood_data[ood_indptr[x] : ood_indptr[x + 1]]
        x_indptr_start = ood_indptr[x]
        for offset in range(len(x_friends)):
            y = x_friends[offset]
            y_friends = ood_indices[ood_indptr[y] : ood_indptr[y + 1]]
            as_close_to_x_as_y = x_friends[x_ranks <= x_ranks[offset]]
            numerator = len(np.intersect1d(as_close_to_x_as_y, y_friends))
            denominator = (
                len(x_friends)
                + len(y_friends)
                - len(np.intersect1d(x_friends, y_friends))
            )
            cohesion[x_indptr_start + offset] = numerator / denominator
            progress_bar.update()
    return cohesion


def paknnld(ood, verbose=False):
    ood = sp.csr_array(ood)
    similarity = ood.copy()
    with ProgressBar(
        total=len(similarity.data),
        disable=not verbose,
    ) as progress_bar:
        similarity.data = cohesion(
            similarity.indptr, similarity.indices, similarity.data, progress_bar
        )
    similarity = similarity.minimum(similarity.transpose()).tocsr()
    return similarity
