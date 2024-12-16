import numpy as np
import scipy
from pnm_mctools import NumericalDifferentiation as nd


def test_dense():
    size = 50
    dc = 1e-6

    c = np.ones((size, 3), dtype=float)

    J_0 = np.arange(1., c.size+1, dtype=float)
    J_0 = np.tile(J_0, reps=[c.size, 1])
    J_0 += (np.arange(c.size) * c.size).reshape((-1, 1))

    def Defect(c, *args):
        return np.matmul(J_0, c.reshape((c.size, 1)))

    J, G = nd.conduct_numerical_differentiation(c, defect_func=Defect, type='full', dc=dc)
    err = np.max(np.abs((J-J_0)/J_0))
    assert err < dc


def test_lowmem():
    size = 50
    dc = 1e-6

    c = np.ones((size, 3), dtype=float)

    J_0 = np.arange(1., c.size+1, dtype=float)
    J_0 = np.tile(J_0, reps=[c.size, 1])
    J_0 += (np.arange(c.size) * c.size).reshape((-1, 1))

    def Defect(c, *args):
        return np.matmul(J_0, c.reshape((c.size, 1)))

    J, G = nd.conduct_numerical_differentiation(c, defect_func=Defect, type='low_mem', dc=dc)
    err = np.max(np.abs((J-J_0)/J_0))
    assert err < dc


def test_locally_constrained():
    size = 50
    dc = 1e-6

    c = np.ones((size, 3), dtype=float)

    rows = np.arange(0, c.size, dtype=int).reshape((-1, 1))
    cols = np.arange(0, c.size, dtype=int).reshape((-1, c.shape[1]))
    rows = np.tile(rows, reps=[1, c.shape[1]])
    cols = np.tile(cols, reps=[1, c.shape[1]])
    rows = rows.flatten()
    cols = cols.flatten()
    data = np.arange(1., rows.size + 1, dtype=float)
    J_orig = scipy.sparse.coo_matrix((data, (rows, cols))).todense()

    J_0 = scipy.sparse.csr_matrix(J_orig)

    def Defect(c, *args):
        return J_0 * c.reshape((c.size, 1))

    J, G = nd.conduct_numerical_differentiation(c, defect_func=Defect, type='constrained', dc=dc)

    J_dense = J.todense()

    mask_zeros = J_orig != 0.
    assert np.all(J_dense[~mask_zeros] == J_orig[~mask_zeros])
    err = np.max(np.abs((J_dense[mask_zeros]-J_orig[mask_zeros])/J_orig[mask_zeros]))
    assert err < dc
