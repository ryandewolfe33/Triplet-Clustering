import scipy.sparse as sp
import numpy as np
import pynndescent
from numba import njit, prange
from numba.typed import List
from numba.types import UniTuple, int32
from numba_progress import ProgressBar


@njit(nogil=True)
def csc_lookup(indptr, indices, data, i, j):
    col = indices[indptr[j] : indptr[j + 1]]
    offset = np.searchsorted(col, i)
    if col[offset] == i:
        return data[indptr[j] + offset]
    return 0


@njit(nogil=True, parallel=True)
def in_sway(
    ood_indptr,
    ood_indices,
    ood_data,
    mutual_friends_row,
    mutual_friends_col,
    progress_bar,
):
    # Linkage graph is same vertices, non-negative integer edge weights
    n_mutual_friends = len(mutual_friends_row)
    row_ind = np.empty(n_mutual_friends, dtype="int32")
    col_ind = np.empty(n_mutual_friends, dtype="int32")
    data = np.empty(n_mutual_friends, dtype="int32")

    for i in prange(n_mutual_friends):
        x, z = mutual_friends_row[i], mutual_friends_col[i]
        y_ranks_x = set(ood_indices[ood_indptr[x] : ood_indptr[x + 1]])
        y_ranks_z = set(ood_indices[ood_indptr[z] : ood_indptr[z + 1]])
        ranks_either = len(y_ranks_x.union(y_ranks_z))

        xz = csc_lookup(ood_indptr, ood_indices, ood_data, x, z)
        for y in y_ranks_x:
            xy = csc_lookup(ood_indptr, ood_indices, ood_data, x, y)
            if xz < xy:  # Remove if xy does not beat xz
                y_ranks_x.remove(y)

        zx = csc_lookup(ood_indptr, ood_indices, ood_data, z, x)
        for y in y_ranks_z:
            zy = csc_lookup(ood_indptr, ood_indices, ood_data, z, y)
            if zx < zy:  # Remove if zy does not beat zx
                y_ranks_z.remove(y)

        in_sway = ranks_either - len(y_ranks_x.union(y_ranks_z))
        row_ind[i] = x
        col_ind[i] = z
        data[i] = in_sway
        progress_bar.update()

    return row_ind, col_ind, data


def rbl(ood: sp.sparray, verbose=False):
    ood = sp.csr_array(ood)
    print("Making Mutual Friend List") if verbose else None
    mutual_friends = sp.triu(ood * ood.T)
    print("Compute In Sway") if verbose else None
    with ProgressBar(
        total=len(mutual_friends.row),
        disable=not verbose,
    ) as progress_bar:
        ood_csc = ood.tocsc()
        row_ind, col_ind, data = in_sway(
            ood_csc.indptr,
            ood_csc.indices,
            ood_csc.data,
            mutual_friends.row,
            mutual_friends.col,
            progress_bar,
        )
    similarity = sp.coo_array(
        (data, (row_ind, col_ind)), shape=(ood.shape[0], ood.shape[0])
    )
    return similarity
