import numpy as np
import scipy
import scipy.sparse
from typing import Callable
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


def LinearAdsorption(network, dilute: int, adsorbed: int, num_components: int, weight='pore.volume', K_ads=1.):
    r"""
    Computes a matrix to include linear adsorption in the system

    Parameters
    ----------
    network: any
        OpenPNM network with geometrical information
    dilute: int
        dilute component
    adsorbed: int
        adsorbed component
    num_components: int
        number of components in the system
    weight: any
        weigth or list of weights which will be assigned to the affected rows
    K_ads
        equilibrium coefficient between the dilute and adsorbed phase

    Returns
    -------
    CSR-matrix of size [Np*Nc, Np*Nc]

    Notes
    -----
    This implementation assumes that the dilute and adsorbed species are two separate components and
    generates the equilibrium by a pseudo-reaction term. The pseudo reaction rate is formulated as:
        r = k * (c*  - c)
    with the equilibrium concentration dependent on the equilibrium constant:
        K = c_ads / c*  --> c* = K / c_ads
    """
    Nc = num_components
    r_ads_0 = np.zeros((network.Np, 2), dtype=float)
    r_ads_1 = np.zeros_like(r_ads_0)
    r_ads_0[:, 0], r_ads_0[:, 1] = -K_ads, 1.
    r_ads_1[:, 0], r_ads_1[:, 1] = K_ads, -1.

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


def AdsorptionSingleComponent(network, c, num_components: int, dilute: int, adsorbed: int, y_max, Vp, a_v, K: Callable, k_r: float = 1e6, type: str = 'Jacobian'):

    A, b = None, None
    Np = network.Np
    Nc = num_components
    V_pore = (np.ones((Np, 1), dtype=float) if Vp is None else Vp.reshape((-1, 1)))
    rows = np.arange(dilute, Np * Nc, Nc, dtype=int).reshape((-1))
    delta_n = adsorbed - dilute

    def Defect(c):
        c_f = c[:, dilute].reshape((-1, 1))
        c_ads = c[:, adsorbed].reshape((-1, 1))
        theta = c_ads / y_max
        theta_eq = K(c_f, c_ads)
        r_ads = k_r * y_max * (theta_eq - theta)
        r_f = -a_v * r_ads
        G = np.zeros((Np * Nc, 1), dtype=float)
        G[rows, :] = r_f
        G[rows + delta_n, :] = r_ads
        return G

    if type == 'Defect':
        b = Defect(c)
    else:
        A, b = NumericalDifferentiation.NumericalDifferentiation(c, defect_func=Defect)

    V_pore = np.hstack([V_pore for _ in range(num_components)]).reshape((-1, 1))
    b *= V_pore
    if A is not None:
        A = A.multiply(V_pore)

    return A, b


class Network(dict):
    def __init__(self):
        self.Np = 10


# network = Network()
# network['pore.volume'] = np.ones((network.Np))

# ads = 0
# dil = 1
# a_V = np.ones((network.Np, 2), dtype=float)
# a_V[:, ads] = 0.1

# # A = LinearAdsorption(network=network, dilute=1, adsorbed=0, num_components=3, K_ads=0.5, weight=['pore.volume', a_V])

# c = np.zeros((network.Np, 2))
# c[:, 0] = 1.

# def K(c_f, c_ads):
#     return c_f * 0.1

# J_dt = scipy.sparse.eye(network.Np*2)
# x = c.reshape((-1, 1))
# x_prev = x.copy()
# for _ in range(10):
#     J_ads, G_ads = AdsorptionSingleComponent(network=network, c=c, dilute=0, adsorbed=1, num_components=2, y_max=1, Vp=None, a_v=1, K=K)
#     J = J_dt + J_ads
#     G = J_dt * (x - x_prev) + G_ads
#     dx = scipy.sparse.linalg.spsolve(J, -G)
#     x += dx.reshape(x.shape)
#     c = x.reshape(c.shape)

# print('finished')
