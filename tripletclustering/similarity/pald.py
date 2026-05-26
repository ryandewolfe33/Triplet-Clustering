import numpy as np
import scipy.sparse as sp
from numba import njit, prange
from numba_progress import ProgressBar


#TODO check parallel
@njit(nogil=True, fastmath=True)
def cohesion(D, progress_bar):
    """
    Compute the C[x, w] cohesion matrix. Algorithm taken from
    https://www.pnas.org/doi/suppl/10.1073/pnas.2003634119#supplementary-materials
    """
    n = D.shape[0]
    C = np.zeros_like(D, dtype="float32")
    for i in prange(n):
        for j in prange(i):
            wi = 0
            n_uij = 0
            for k in range(n):
                if D[i, k] <= D[i, j] or D[j, k] <= D[i, j]:
                    n_uij += 1
            # assert n_uij > 0
            for k in range(n):
                if D[i, k] <= D[i, j] or D[j, k] <= D[i, j]:
                    if D[i, k] < D[j, k]:
                        C[i, k] += 1 / n_uij
                    elif D[j, k] < D[i, k]:
                        C[j, k] += 1 / n_uij
                    else:
                        C[i, k] += 0.5 / n_uij
                        C[j, k] += 0.5 / n_uij
            progress_bar.update()
    C = C / (n - 1)
    return C


def pald(ood, verbose=False):
    n_pairs = ood.shape[0] * (ood.shape[0] - 1) // 2
    with ProgressBar(
        total=n_pairs,
        disable=not verbose,
    ) as progress_bar:
        similarity = cohesion(ood, progress_bar)
    return similarity
