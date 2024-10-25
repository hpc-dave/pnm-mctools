from typing import List
import numpy as np
import scipy
from . import Operators as ops


def upwind(network,
           fluxes,
           include: int | List[int] | None = None,
           exclude: int | List[int] | None = None,
           Nc: int | None = None) -> scipy.sparse.csr_matrix:
    r"""
    Constructs a [Nt, Np] matrix representing a directed network based on the upwind
    fluxes

    Parameters
    ----------
    fluxes: any
        fluxes which determine the upwind direction, see below for more details
    include: list
        a list of integers to specify for which components the matrix should be constructed,
        for components which are not listed here, the rows will be 0. If 'None' is provided,
        all components will be selected
    exclude: list
        Inverse of include, without effect if include is specified
    Nc: int|None
        number of components in the system
    Returns
    -------
    A [Nt, Np] sized CSR-matrix representing a directed network

    Notes
    -----
    The direction of the fluxes is directly linked with the storage of the connections
    inside the OpenPNM network. For more details, refer to the 'create_incidence_matrix' method
    of the network module.
    The resulting matrix IS NOT SCALED with the fluxes and can also be used for determining
    upwind interpolated values.
    The provided fluxes can either be:
        int/float - single value
        list/numpy.ndarray - with size num_components applies the values to each component separately
        numpy.ndarray - with size Nt applies the fluxes to each component by throat: great for convection
        numpy.ndarray - with size Nt * num_components is the most specific application for complex
                        multicomponent coupling, where fluxes can be opposed to each other within
                        the same throat
    """
    net, Np, Nt, Nc, include = ops.unpack_network(network=network, Nc=Nc, include=include, exclude=exclude)

    # Note, that this function can be shortened significantly. However, I decided to leave it this way, since
    # the most general option is quite difficult to understand on the first glance and having this longer version
    # is beneficial for the understanding of (new) readers. Since the performance penalty of those maximum
    # 5 if-statements should be negligible under regular conditions, I decided to keep it this way.
    if Nc == 1:
        # check input
        if isinstance(fluxes, float) or isinstance(fluxes, int):
            _fluxes = np.zeros((Nt)) + fluxes
        elif fluxes.size == Nt:
            _fluxes = fluxes
        else:
            raise ValueError('invalid flux dimensions')
        weights = np.append((_fluxes < 0).astype(float), _fluxes > 0)
        return np.transpose(net.create_incidence_matrix(weights=weights, fmt='csr'))
    else:
        if include is None:
            include = range(Nc)
        num_included = len(include)

        im = np.transpose(net.create_incidence_matrix(fmt='coo'))

        data = np.zeros((im.data.size, num_included), dtype=float)
        rows = np.zeros((im.data.size, num_included), dtype=int)
        cols = np.zeros((im.data.size, num_included), dtype=int)

        pos = 0
        for n in include:
            rows[:, pos] = im.row * Nc + n
            cols[:, pos] = im.col * Nc + n
            data[:, pos] = im.data
            pos += 1

        if isinstance(fluxes, float) or isinstance(fluxes, int):
            # single provided value
            _fluxes = np.zeros((Nt)) + fluxes
            weights = np.append(_fluxes < 0, _fluxes > 0)
            pos = 0
            for n in include:
                data[:, pos] = weights
                pos += 1
        elif (isinstance(fluxes, list) and len(fluxes) == Nc)\
                or (isinstance(fluxes, np.ndarray) and fluxes.size == Nc):
            # a list of values for each component
            _fluxes = np.zeros((Nt))
            pos = 0
            for n in include:
                _fluxes[:] = fluxes[n]
                weights = np.append(_fluxes < 0, _fluxes > 0)
                data[:, pos] = weights
                pos += 1
        elif fluxes.size == Nt:
            # fluxes for each throat, e.g. for single component or same convective fluxes
            # for each component
            weights = np.append(fluxes < 0, fluxes > 0)
            pos = 0
            for n in include:
                data[:, pos] = weights.reshape((Nt*2))
                pos += 1
        elif (len(fluxes.shape)) == 2\
            and (fluxes.shape[0] == Nt)\
                and (fluxes.shape[1] == Nc):
            # each throat has different fluxes for each component
            pos = 0
            for n in include:
                weights = np.append(fluxes[:, n] < 0, fluxes[:, n] > 0)
                data[:, pos] = weights.reshape((Nt*2))
                pos += 1
        else:
            raise ValueError('fluxes have incompatible dimension')

        rows = np.ndarray.flatten(rows)
        cols = np.ndarray.flatten(cols)
        data = np.ndarray.flatten(data)
        mat_shape = (Nt * Nc, Np * Nc)
        upwind = scipy.sparse.coo_matrix((data, (rows, cols)), shape=mat_shape)
        return scipy.sparse.csr_matrix(upwind)


def central_difference(network,
                       include: int | List[int] | None = None,
                       exclude: int | List[int] | None = None,
                       Nc: int | None = None):
    r"""
    Constructs a [Nt, Np] matrix for the interpolation of values at the throats
    from pore values

    Parameters
    ----------
    include: list
        a list of integers to specify for which components the matrix should be constructed,
        for components which are not listed here, the rows will be 0. If 'None' is provided,
        all components will be selected
    exclude: list
        Inverse of include, without effect if include is specified
    Returns
    -------
    A [Nt, Np] sized CSR-matrix representing a directed network

    """
    net, Np, Nt, Nc, include = ops.unpack_network(network=network, Nc=Nc, include=include, exclude=exclude)

    # in principle the first if-statement is not required, but we can save a bit of allocation hustling
    # and matrix conversion on the way, so I keep it here
    if Nc == 1:
        weights = np.full((2 * Nt), fill_value=0.5, dtype=float)
        return np.transpose(net.create_incidence_matrix(weights=weights, fmt='csr'))
    else:
        if include is None:
            include = range(Nc)
        num_included = len(include)

        im = np.transpose(net.create_incidence_matrix(fmt='coo'))

        data = np.zeros((im.data.size, num_included), dtype=float)
        rows = np.zeros((im.data.size, num_included), dtype=int)
        cols = np.zeros((im.data.size, num_included), dtype=int)

        pos = 0
        for n in include:
            rows[:, pos] = im.row * Nc + n
            cols[:, pos] = im.col * Nc + n
            data[:, pos] = 0.5
            pos += 1

        rows = np.ndarray.flatten(rows)
        cols = np.ndarray.flatten(cols)
        data = np.ndarray.flatten(data)
        mat_shape = (Nt * Nc, Np * Nc)
        cds = scipy.sparse.coo_matrix((data, (rows, cols)), shape=mat_shape)
        return scipy.sparse.csr_matrix(cds)
