import pytest
import scipy
import numpy as np
import openpnm as op
from pnm_mctools import Reactions


@pytest.fixture
def network():
    network = op.network.Cubic(shape=(10, 10, 10))
    network.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
    network.regenerate_models()
    return network


def test_LinearReaction_source(network):
    num_components = 3
    k = 2.
    comp = 1
    weight = 'pore.volume'

    A = Reactions.LinearReaction(network,
                                 num_components=num_components,
                                 k=k,
                                 component=comp,
                                 weight=weight)

    num_pores = network.num_pores()
    assert A.shape[0] == A.shape[1], 'the returned matrix is not square'
    assert A.shape[0] == num_pores * num_components
    A_coo = scipy.sparse.coo_matrix(A)
    assert A_coo.nnz == num_pores
    assert np.all(A_coo.col == A_coo.coords)
    assert A_coo.col[0] == np.array(range(comp, num_components, num_pores * num_components))
    alpha = A_coo.data
    alpha_c = -k * network['pore.volume']
    assert np.all(alpha == alpha_c)


def test_LinearReaction_sink_source(network):
    num_components = 3
    k = 2.
    educt = 1
    product = 2
    weight = 'pore.volume'

    A = Reactions.LinearReaction(network,
                                 num_components=num_components,
                                 k=k,
                                 educt=educt, product=product,
                                 weight=weight)

    num_pores = network.num_pores()
    assert A.shape[0] == A.shape[1], 'the returned matrix is not square'
    assert A.shape[0] == num_pores * num_components, 'the matrix size is wrong'
    A_coo = scipy.sparse.coo_matrix(A)
    assert A_coo.nnz == num_pores*2, 'number of non-zeros is wrong'
    rows = np.array(range(0, num_components * num_pores, num_components)).reshape((-1, 1))
    cols = np.copy(rows)
    rows = np.hstack([rows, rows+1, rows+2]).reshape((-1))
    cols = np.hstack([cols, cols+1, cols+1]).reshape((-1))
    data = np.asarray([0., k, -k]).reshape((1, -1)) * network['pore.volume'].reshape(-1, 1)
    data = data.reshape((-1))

    A_comp = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(num_pores*num_components, num_pores*num_components))
    A_comp.eliminate_zeros()

    err_max = np.max(np.abs((A_comp - A_coo)))
    assert err_max == 0.0


def test_LinearReaction_sink_source_multiple(network):
    num_components = 3
    k = 2.
    educt = 1
    product = [0, 2]
    weight = 'pore.volume'

    A = Reactions.LinearReaction(network,
                                 num_components=num_components,
                                 k=k,
                                 educt=educt, product=product,
                                 weight=weight)

    num_pores = network.num_pores()
    assert A.shape[0] == A.shape[1], 'the returned matrix is not square'
    assert A.shape[0] == num_pores * num_components, 'the matrix size is wrong'
    A_coo = scipy.sparse.coo_matrix(A)
    assert A_coo.nnz == num_pores * 3, 'number of non-zeros is wrong'

    rows = np.array(range(0, num_components * num_pores, num_components)).reshape((-1, 1))
    cols = np.copy(rows)
    rows = np.hstack([rows, rows+1, rows+2]).reshape((-1))
    cols = np.hstack([cols+1, cols+1, cols+1]).reshape((-1))
    data = np.asarray([-k, k, -k]).reshape((1, -1)) * network['pore.volume'].reshape(-1, 1)
    data = data.reshape((-1))

    A_comp = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(num_pores*num_components, num_pores*num_components))
    A_comp.eliminate_zeros()

    err_max = np.max(np.abs((A_comp - A_coo)))

    assert err_max == 0.0


def test_LinearReaction_check_ids(network):
    num_components = 2
    k = 2.
    educt = 2
    product = None
    with pytest.raises(ValueError):
        Reactions.LinearReaction(network,
                                 num_components=num_components,
                                 k=k,
                                 educt=educt, product=product)
    educt = 1
    product = [2]
    with pytest.raises(ValueError):
        Reactions.LinearReaction(network,
                                 num_components=num_components,
                                 k=k,
                                 educt=educt, product=product)
    educt = -1
    product = [1]
    with pytest.raises(ValueError):
        Reactions.LinearReaction(network,
                                 num_components=num_components,
                                 k=k,
                                 educt=educt, product=product)
    educt = 1
    product = [1]
    with pytest.raises(ValueError):
        Reactions.LinearReaction(network,
                                 num_components=num_components,
                                 k=k,
                                 educt=educt, product=product)
