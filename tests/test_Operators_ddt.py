import pytest
import numpy as np
import scipy
import openpnm as op
from pnm_mctools import Operators as ops


@pytest.fixture
def network():
    network = op.network.Cubic(shape=(10, 10, 10))
    network.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
    network.regenerate_models()
    return network


def test_ddt_componentarray():
    Np, Nc = 100, 2
    dt = 1.5
    c = np.zeros((Np, Nc), dtype=float)
    weight = np.array(range(1, Np+1), dtype=float).reshape((-1))

    ddt = ops.ddt(c, dt=dt, weight=weight)
    ddt_coo = scipy.sparse.coo_matrix(ddt)

    rows = np.array(range(Np*Nc))
    cols = rows
    data = np.tile(weight.reshape((-1, 1)), reps=(1, Nc)).reshape((-1)) / dt

    ddt_comp = scipy.sparse.coo_matrix((data, (rows, cols)))

    err_max = np.max(ddt_coo - ddt_comp)
    assert err_max == 0.


def test_ddt_network(network):

    Np = network.num_pores()
    Nc = 2
    dt = 1.5
    weight = 'pore.volume'
    ddt = ops.ddt(network, dt=dt, weight=weight, Nc=Nc)
    ddt_coo = scipy.sparse.coo_matrix(ddt)

    weight = network[weight]
    rows = np.array(range(Np*Nc))
    cols = rows
    data = np.tile(weight.reshape((-1, 1)), reps=(1, Nc)).reshape((-1)) / dt

    ddt_comp = scipy.sparse.coo_matrix((data, (rows, cols)))

    err_max = np.max(ddt_coo - ddt_comp)
    assert err_max == 0.


def test_ddt_include(network):
    Np = network.num_pores()
    Nc = 2
    dt = 1.5
    include = 0
    weight = 'pore.volume'
    ddt = ops.ddt(network, dt=dt, weight=weight, Nc=Nc, include=include)
    ddt_coo = scipy.sparse.coo_matrix(ddt)

    weight = network[weight]
    rows = np.array(range(Np*Nc))
    cols = rows
    data = np.tile(weight.reshape((-1, 1)), reps=(1, Nc)) / dt
    for n in range(Nc):
        if n == include:
            continue
        data[:, n] = 0.
    data = data.reshape((-1))

    ddt_comp = scipy.sparse.coo_matrix((data, (rows, cols)))

    err_max = np.max(ddt_coo - ddt_comp)
    assert err_max == 0.


def test_ddt_network_custom_weight(network):

    Np = network.num_pores()
    Nc = 2
    dt = 1.5
    weight = -0.33
    weight = np.full((Np), fill_value=weight)
    ddt = ops.ddt(network, dt=dt, weight=weight, Nc=Nc)
    ddt_coo = scipy.sparse.coo_matrix(ddt)

    rows = np.array(range(Np*Nc))
    cols = rows
    data = np.tile(weight.reshape((-1, 1)), reps=(1, Nc)).reshape((-1)) / dt

    ddt_comp = scipy.sparse.coo_matrix((data, (rows, cols)))

    err_max = np.max(ddt_coo - ddt_comp)
    assert err_max == 0.


def test_ddt_exceptions(network):
    dt = 1.5
    Nc = 2
    weight = 'pore.volume'
    with pytest.raises(ValueError):
        ops.ddt(dt=dt, Nc=Nc, weight=weight)

    with pytest.raises(ValueError):
        ops.ddt(network=network, dt=-0.1, Nc=Nc, weight=weight)

    with pytest.raises(ValueError):
        Np = 5
        c = np.zeros((Np, Nc), dtype=float)
        ops.ddt(c=c, network=network, dt=dt, Nc=Nc, weight=weight)
