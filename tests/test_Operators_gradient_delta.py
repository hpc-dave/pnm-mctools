import pytest
import numpy as np
import scipy
import openpnm as op
from pnm_mctools import Operators as ops
from pnm_mctools import ToolSet as ts


@pytest.fixture
def network():
    network = op.network.Cubic(shape=(2, 2, 2))
    network.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
    network.regenerate_models()
    return network


def test_gradient(network):
    Np = network.num_pores()
    Nt = network.num_throats()
    Nc = 2
    l_conduits = np.array(range(1, Nt + 1), dtype=float)
    mt = ts.MulticomponentTools(network, num_components=Nc)

    grad = ops.gradient(mt, conduit_length=l_conduits)

    assert grad.shape[0] == Nt * Nc
    assert grad.shape[1] == Np * Nc

    conns = network['throat.conns']
    rows = np.zeros((Nt, Nc, 2), dtype=int)
    cols = np.zeros_like(rows)
    data = np.zeros_like(rows, dtype=float)

    for i in range(Nt):
        for n in range(Nc):
            rows[i, n, 0], rows[i, n, 1] = i * Nc + n, i * Nc + n
            cols[i, n, 0], cols[i, n, 1] = conns[i, 0] * Nc + n, conns[i, 1] * Nc + n
            data[i, n, 0], data[i, n, 1] = -1./l_conduits[i], 1./l_conduits[i]

    grad_comp = scipy.sparse.coo_matrix((data.flatten(), (rows.flatten(), cols.flatten())))
    grad_coo = scipy.sparse.coo_matrix(grad)
    err_max = np.max(np.abs(grad_coo - grad_comp))

    assert err_max == 0.


def test_delta(network):
    Np = network.num_pores()
    Nt = network.num_throats()
    Nc = 2

    mt = ts.MulticomponentTools(network, num_components=Nc)

    grad = ops.delta(mt)

    assert grad.shape[0] == Nt * Nc
    assert grad.shape[1] == Np * Nc

    conns = network['throat.conns']
    rows = np.zeros((Nt, Nc, 2), dtype=int)
    cols = np.zeros_like(rows)
    data = np.zeros_like(rows, dtype=float)

    for i in range(Nt):
        for n in range(Nc):
            rows[i, n, 0], rows[i, n, 1] = i * Nc + n, i * Nc + n
            cols[i, n, 0], cols[i, n, 1] = conns[i, 0] * Nc + n, conns[i, 1] * Nc + n
            data[i, n, 0], data[i, n, 1] = -1., 1.

    grad_comp = scipy.sparse.coo_matrix((data.flatten(), (rows.flatten(), cols.flatten())))
    grad_coo = scipy.sparse.coo_matrix(grad)
    err_max = np.max(np.abs(grad_coo - grad_comp))

    assert err_max == 0.
