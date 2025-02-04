import pytest
import numpy as np
import scipy
import openpnm as op
from pnm_mctools import Operators as ops
from pnm_mctools import ToolSet as ts


@pytest.fixture
def network_pseudo1D():
    network = op.network.Cubic(shape=(10, 1, 1))
    network.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
    network.regenerate_models()
    # quick check that all throats are numbered upwards starting from x=0
    # quite important for the further evaluation
    conns = network['throat.conns']
    x_coords = network['pore.coords'][:, 0]
    x_pos = x_coords[conns.reshape((-1))].reshape((-1, 2))
    x_pos = np.min(x_pos, axis=1)
    assert np.all(x_pos[:-1] < x_pos[1:])
    return network


@pytest.fixture
def network():
    network = op.network.Cubic(shape=(10, 10, 10))
    network.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
    network.regenerate_models()
    return network


def test_sum_pseudo1D_build(network_pseudo1D):
    Np = network_pseudo1D.num_pores()
    Nt = network_pseudo1D.num_throats()
    Nc = 2

    sum = ops.sum(network=network_pseudo1D, Nc=Nc)

    assert isinstance(sum, ts.SumObject), 'wrong object type returned'

    assert sum.matrix.shape[0] == Np*Nc, 'number of matrix rows is wrong'
    assert sum.matrix.shape[1] == Nt*Nc, 'number of matrix columns is wrong'

    sum_coo = scipy.sparse.coo_matrix(sum.matrix)

    data = np.ones((Nt*Nc))
    sum_comp = scipy.sparse.diags([data, -data], offsets=[0, -2], shape=(Np*Nc, Nt*Nc), format='coo', dtype=float)

    err_max = np.max(np.abs(sum_comp - sum_coo))

    assert err_max == 0.0


def test_sum_with_rates(network):
    Np = network.num_pores()
    Nt = network.num_throats()
    Nc = 2

    sum = ops.sum(network=network, Nc=Nc)

    assert isinstance(sum, ts.SumObject), 'wrong object type returned'

    rates = np.array(range(Nt), dtype=float).reshape((-1, 1))
    rates = np.hstack([rates, rates + 0.25])

    sum_rates = sum(rates.reshape((-1)))
    sum_rates = sum_rates.reshape((-1, Nc))

    conns = network['throat.conns']
    sum_comp = np.zeros((Np, Nc), dtype=float)
    for n in range(Nc):
        for i in range(Np):
            sum_comp[i, n] = np.sum(rates[conns[:, 0] == i, n]) - np.sum(rates[conns[:, 1] == i, n])

    err_max = np.max(np.abs(sum_rates-sum_comp))
    assert err_max == 0.


def test_sum_exclude(network):
    Nc = 3

    mt = ts.MulticomponentTools(network, num_components=Nc)

    exclude = [1]

    sum = ops.sum(mt, exclude=exclude)

    rows = np.array(range(exclude[0], network.num_pores()*Nc, Nc))
    mask = np.zeros((network.num_pores()*Nc), dtype=bool)
    mask[rows] = True

    assert sum.matrix[mask, :].nnz == 0
