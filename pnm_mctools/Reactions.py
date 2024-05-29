import numpy as np
import scipy
import scipy.sparse
try:
    from . import NumericalDifferentiation
except ImportError:
    import NumericalDifferentiation


def LinearReaction(network, num_components: int, k, educt: int, product=None, weight='pore.volume'):
    r"""
    A convenience function to provide an implicit source term based on a linear reaction

    Parameters
    ----------
    network: any
        An openpnm network with geometric information
    num_components: int
        number of components in the system
    educt: int
       component ID which serves as educt
    product: list
        list of component IDs which serve as products, if an integer is provided it will be converted into a list
    k: any
        constant reaction rate coefficient, can be single value or array of size [Np,]
    weight: any
        pore based factor for multiplication, by default the pore volume is used, can be a list of weights

    Returns
    -------
    CSR - Matrix of size [Np*Nc, Np*Nc]

    Notes
    -----
    Following reaction is assumed to take place:
          k
        E -> P_1 + P_2
    so that the reaction rate becomes:
        r = k * c_E
    """
    if product is None:
        product = []
    if isinstance(product, int):
        product = [product]
    if k is None:
        raise ValueError('the rate coefficient is not specified')
    if not isinstance(product, list) or not isinstance(product[0], int):
        raise TypeError('educt either has to be provided as int or list of ints')

    if np.any(np.asarray(educt) >= num_components) or np.any(np.asarray(product) >= num_components):
        raise ValueError('at least one educt or product ID is out of range')
    if np.any(np.asarray(educt) < 0) or np.any(np.asarray(product) < 0):
        raise ValueError('at least one educt or product ID is below 0!')
    if educt in product:
        raise ValueError('Overlap in product and educt specification')

    num_pores = network.Np
    aff_species = [educt] + product
    num_species_aff = len(aff_species)

    rows = np.zeros((num_pores * num_species_aff, 1), dtype=int)
    cols = np.zeros_like(rows)
    values = np.zeros_like(rows, dtype=float)

    block_start_row = np.asarray(np.arange(0, num_pores*num_components, num_components)).reshape((-1, 1))

    for n, id in enumerate(aff_species):
        ind = range(n, rows.shape[0], num_species_aff)
        rows[ind] = block_start_row + id
        cols[ind] = (block_start_row + educt)
        values[ind] = -k if id == educt else k

    if weight is not None:
        weights = [weight] if not isinstance(weight, list) else weight
        for w in weights:
            v = network[w].copy() if isinstance(w, str) else w
            if isinstance(v, float) or isinstance(v, int):
                values *= v
            elif isinstance(v, np.ndarray) and v.size == num_pores:
                values *= np.tile(v, (1, num_species_aff)).reshape((-1, 1))
            elif isinstance(v, np.ndarray) and v.size == values.size:
                values *= v.reshape((-1, 1))
            else:
                raise TypeError('Cannot use this weight type')

    rows = rows.ravel()
    cols = cols.ravel()
    values = values.ravel()

    num_rows = num_pores * num_components
    A = scipy.sparse.coo_matrix((values, (rows, cols)), shape=(num_rows, num_rows))
    return A.tocsr()


def LinearAdsorption(network, dilute, adsorbed, num_components: int, weight='pore.volume', K_ads=1., k_reac=1.):
    Nc = num_components
    r_ads_0 = np.zeros((network.Np, 2), dtype=float)
    r_ads_1 = np.zeros_like(r_ads_0)
    r_ads_0[:, 0], r_ads_0[:, 1] = -k_reac * K_ads, k_reac
    r_ads_1[:, 0], r_ads_1[:, 1] = k_reac * K_ads, -k_reac

    if weight is not None:
        weights = weight if isinstance(weight, list) else [weight]
        for w in weights:
            w_loc = network[w] if isinstance(w, str) else w
            if isinstance(w_loc, float) or isinstance(w_loc, int):
                r_ads_0 *= w_loc
                r_ads_1 *= w_loc
            elif isinstance(w_loc, np.ndarray) and w_loc.size == network.Np:
                r_ads_0 *= w_loc.reshape((-1, 1))
                r_ads_1 *= w_loc.reshape((-1, 1))
            elif isinstance(w_loc, np.ndarray) and w_loc.ndim == 2 and w_loc.shape[1] == 2:
                r_ads_0 *= w_loc[:, 0].reshape((-1, 1))
                r_ads_1 *= w_loc[:, 1].reshape((-1, 1))
            else:
                raise TypeError('Incompatible weight provided')

    row = np.arange(dilute, network.Np * Nc, Nc, dtype=int).reshape((-1, 1))
    col = row.copy()
    delta_n = adsorbed - dilute
    row = np.hstack((row, row, row+delta_n, row+delta_n)).flatten()
    col = np.hstack((col, col+delta_n, col, col+delta_n)).flatten()
    data = np.hstack((r_ads_0, r_ads_1)).flatten()
    A = scipy.sparse.coo_matrix((data, (row, col)), shape=(network.Np*Nc, network.Np*Nc))
    return A.tocsr()


# class Network(dict):
#     def __init__(self):
#         self.Np = 10


# network = Network()
# network['pore.volume'] = np.ones((network.Np))

# ads = 0
# dil = 1
# a_V = np.ones((network.Np, 2), dtype=float)
# a_V[:, ads] = 0.1

# A = LinearAdsorption(network=network, dilute=1, adsorbed=0, num_components=2, K_ads=0.5, weight=['pore.volume', a_V])
