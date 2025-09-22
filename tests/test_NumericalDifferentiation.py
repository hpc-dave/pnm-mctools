import numpy as np
import scipy
import openpnm as op
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


def test_sparsity_exploit():
    # test sparsity exploiting version
    shape = [10, 10, 10]

    Nc = 3
    dc = 1e-6
    c = np.ones((np.prod(shape), Nc), dtype=float)
    pn = op.network.Cubic(shape=shape, spacing=1)
    J_0 = pn.create_adjacency_matrix(weights=pn['throat.conns']+1, fmt='coo')
    rows = np.hstack([J_0.row*Nc + n for n in range(Nc)])
    cols = np.hstack([J_0.col*Nc + n for n in range(Nc)])
    data = np.hstack([J_0.data * J_0.size * (n+1) for n in range(Nc)])
    J_0 = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(c.size, c.size), dtype=float)
    J_0 += scipy.sparse.spdiags([np.arange(1, J_0.shape[0]+1)], [0], format='csr')

    def Defect(c):
        return J_0 * c.reshape((-1, 1))

    opt = {}
    J, G = nd.conduct_numerical_differentiation(c, defect_func=Defect, type='full', dc=dc, network=pn, opt=opt)
    if J.nnz == J_0.nnz and np.all(J.indices == J_0.indices) and np.all(J.indptr == J_0.indptr):  # noqa: E501
        err = np.max((J.data/J_0.data)-1)
    else:
        err = np.inf
    assert err < 1e-3
    assert opt, 'the optimization dictionary is empty'

    J, G = nd.conduct_numerical_differentiation(c, defect_func=Defect, type='full', dc=dc, network=pn, opt=opt)
    if J.nnz == J_0.nnz and np.all(J.indices == J_0.indices) and np.all(J.indptr == J_0.indptr):  # noqa: E501
        err = np.max((J.data/J_0.data)-1)
    else:
        err = np.inf
    assert err < 1e-3
