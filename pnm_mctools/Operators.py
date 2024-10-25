from typing import List, Tuple, Any
import numpy as np
import scipy
import ToolSet as ts


def _compute_flux_matrix(Nt: int, Nc: int, *args):
    r"""
    computes matrix of size [Np*Nc, Nt*Nc], where all arguments are multiplied with the last argument

    Parameters
    ----------
    Factors to multiply with the final argument, where the final argument is a [Nt*Nc, Np*Nc] matrix

    Returns
    -------
    [Np*Nc, Nt*Nc] sized matrix
    """
    fluxes = args[-1].copy()
    for i in range(len(args)-1):
        arg = args[i]
        if isinstance(arg, list) and len(arg) == Nc:
            fluxes = fluxes.multiply(np.tile(np.asarray(arg), Nt))
        elif isinstance(arg, np.ndarray):
            _arg = np.tile(arg.reshape(-1, 1), reps=(1, Nc)) if arg.size == Nt else arg
            fluxes = fluxes.multiply(_arg.reshape(-1, 1))
        else:
            fluxes = fluxes.multiply(args[i])
    return fluxes


def unpack_network(network, Nc, include, exclude) -> Tuple[Any, int, int, int, Any]:
    if isinstance(network, ts.MulticomponentTools):
        net = network.get_network()
        if Nc is None:
            Nc = network.get_num_components()
    else:
        net = network
        if Nc is None:
            raise ValueError('the number of components is not specified, cannot continue')
    Np = net.num_pores()
    Nt = net.Nt
    include = ts.get_include(include=include, exclude=exclude, Nc=Nc)
    return net, Np, Nt, Nc, include


class SumObject:
    r"""
    A helper object, which acts as a matrix, but also provides convenient overloads
    """
    def __init__(self, matrix, Nc: int, Nt: int):
        r"""
        Initializes the object

        Parameters
        ----------
        matrix:
            A matrix for computing a sum/divergence matrix
        Nc: int
            number of components
        Nt:
            number of throats
        """
        self.matrix = matrix
        self.Nt = Nt
        self.Nc = Nc

    def __mul__(self, other):
        r"""
        multiplication operator

        Parameters
        ----------
        other:
            vector, matrix or scalar to multiply with the internal matrix

        Returns
        -------
        quadratic matrix or vector, depending on the input type
        """
        return self.matrix * other

    def __matmul__(self, other):
        r"""
        matrix multiplication operator

        Parameters
        ----------
        other:
            vector, matrix or scalar to multiply with the internal matrix

        Returns
        -------
        quadratic matrix or vector, depending on the input type
        """
        return self.matrix @ other

    def __call__(self, *args):
        r"""
        calling operator, for convenience to provide high readability of the code

        Parameters
        ----------
        args:
            multiple arguments, which are multiplied with the last instance

        Returns
        -------
        quadratic matrix or vector, depending on the input type
        """
        return self.matrix * _compute_flux_matrix(self.Nt, self.Nc, *args)

    def multiply(self, *args, **kwargs):
        return self.matrix.multiply(*args, **kwargs)


def ddt(c: np.ndarray | None = None,
        network=None,
        dt: float = 1.,
        weight: np.ndarray | str = 'pore.volume',
        include: int | List[int] | None = None,
        exclude: int | List[int] | None = None,
        Nc: int | None = None):
    r"""
    Computes partial time derivative matrix

    Parameters
    ----------
    c: np.ndarray|ts.MulticomponentTools
        array with species or tool with relevant information
    dt: float
        discretized time step size
    weight: np.ndarray|str
        a weight which can be applied to the time derivative, usually that should be
        the volume of the computational cell, the string is only allowed if an instance of
        MulticomponentTools is provided
    include: int|list[int]|None
        an ID or list of IDs which should be included in the matrix, if 'None' is provided,'
        all values will be used
    exclude: int|list[int]|None
        inverse of include, without effect if include is specified
    Returns
    -------
    Matrix in CSR format

    Notes
    -----
    By default, a finite volume discretization is assumed, therefore the standard form of
    the partial derivative is given by

    \iiint \frac{\partial}{\partial t} \mathrm{d}V \approx \frac{\Delta V}{\Delta t}

    Note that here the integrated variable is ommitted in the description, as it will be provided
    either by the solution vector for implicit treatment and by the field for explicit components
    """

    if isinstance(c, np.ndarray):
        Np = c.shape[0]
        if Nc is None:
            Nc = c.shape[1]
        weight_l = weight
    else:
        network = c if network is None else network     # noqa: E501 small hack so the very first argument can be positional for all kinds of objects
        if network is None:
            raise ValueError('the provided network is not specified and the c-array is also not helpful, cannot continue')  # noqa: E501
        net, Np, _, Nc, include = unpack_network(network=network, Nc=Nc, include=include, exclude=exclude)
        if isinstance(weight, str):
            weight_l = net[weight]
        else:
            weight_l = weight

    if dt <= 0.:
        raise ValueError(f'timestep is invalid, following constraints were violated: {dt} !> 0')
    if not isinstance(weight_l, np.ndarray):
        raise ValueError('The provided weight has to be an array!')

    include = ts.get_include(Nc=Nc, include=include, exclude=exclude)

    dVdt = weight_l.copy()
    dVdt /= dt

    dVdt = dVdt.reshape((-1, 1))
    if Nc > 1:
        if dVdt.size == Np:
            dVdt = np.tile(A=dVdt, reps=Nc)
        if include is not None:
            mask = np.asarray([n in include for n in range(Np)], dtype=bool).reshape((1, -1))
            mask = np.tile(A=mask, reps=(Np, 1))
            dVdt[~mask] = 0.
    ddt = scipy.sparse.spdiags(data=[dVdt.ravel()], diags=[0])
    return ddt


def sum(network,
        include: int | List[int] | None = None,
        exclude: int | List[int] | None = None,
        Nc: int | None = None) -> SumObject:
    r"""
    Constructs summation matrix

    Parameters
    ----------
        network
            An instance with similar signatures as a MulticomponentTools or OpenPNM network object
            In the case of an OpenPNM network, the argument Nc has to be specified
        include: int|list[int]|None
            identifier, which components should be included in the divergence, all other
            rows will be set to 0
        exclude: int|list[int]|None
            identifier, which components shall be exluded, respectively which rows shall be
            set to 0. Without effect if include is specified
        Nc: int|None
            number of components that the sum object shall be created for

    Returns
    -------
        Summation matrix

    Notes
    -----
        For the sum, the flux in each throat is assumed to be directed
        according to underlying specification of the throats in the network. More specifically,
        the flux is directed according the to the 'throat.conn' array, from the pore in column 0 to the pore
        in column 1, e.g. if the throat.conn array looks like this:
        [
            [0, 1]
            [1, 2]
            [2, 3]
        ]
        Then the fluxes are directed from pore 0 to 1, 1 to 2 and 2 to 3. A potential network could be:
        (0) -> (1) -> (2) -> (3)
    """

    net, Np, Nt, Nc, include = unpack_network(network=network, Nc=Nc, include=include, exclude=exclude)

    weights = np.ones_like(net['throat.conns'][:, 0], dtype=float)
    weights = np.append(-weights, weights)

    sum_mat = net.create_incidence_matrix(weights=weights, fmt='coo')
    if Nc > 1:
        if include is None:
            include = range(Nc)
        num_included = len(include)

        data = np.zeros((sum_mat.data.size, num_included), dtype=float)
        rows = np.zeros((sum_mat.data.size, num_included), dtype=float)
        cols = np.zeros((sum_mat.data.size, num_included), dtype=float)
        pos = 0
        for n in include:
            rows[:, pos] = sum_mat.row * Nc + n
            cols[:, pos] = sum_mat.col * Nc + n
            data[:, pos] = weights
            pos += 1
        rows = np.ndarray.flatten(rows)
        cols = np.ndarray.flatten(cols)
        data = np.ndarray.flatten(data)
        mat_shape = (Np * Nc, Nt * Nc)
        sum_mat = scipy.sparse.coo_matrix((data, (rows, cols)), shape=mat_shape)

    # converting to CSR format for improved computation
    sum_mat = scipy.sparse.csr_matrix(sum_mat)

    return SumObject(matrix=sum_mat, Nc=Nc, Nt=Nt)


def gradient(network,
             conduit_length: str | np.ndarray | None = None,
             include: int | List[int] | None = None,
             exclude: int | List[int] | None = None,
             Nc: int | None = None) -> scipy.sparse.csr_matrix:
    r"""
    Computes a gradient matrix

    Parameters
    ----------
        network
            An instance with similar signatures as a MulticomponentTools or OpenPNM network object
            In the case of an OpenPNM network, the argument Nc has to be specified
        conduit_length: str|np.ndarray|None
            length of the conduit for computation of the gradient, by default the distance between
            pore centers is utilized
        include: int|[int]|None
            int or list of ints with IDs to include, if 'None' is provided
            all IDs will be included
        exlude: int|[int]|None
            inverse of include, without effect if include is specified
        Nc: int|None
            number of components, needs to be specified if used if network doesn't have the function get_num_components
    Returns
    -------
        a gradient matrix in CSR-format

    Notes
    -----
        The direction of the gradient is given by the connections specified in the network,
        mores specifically from conns[:, 0] to conns[:, 1]
    """
    net, Np, Nt, Nc, include = unpack_network(network=network, Nc=Nc, include=include, exclude=exclude)

    if conduit_length is None:
        conns = net['throat.conns']
        p_coord = net['pore.coords']
        dist = np.sqrt(np.sum((p_coord[conns[:, 0], :] - p_coord[conns[:, 1], :])**2, axis=1))
    elif isinstance(conduit_length, str):
        dist = net[conduit_length]
    elif isinstance(conduit_length, np.ndarray):
        if conduit_length.size != Nt:
            raise ValueError('The size of the conduit_length argument is incompatible with the number of throats!'
                             + f' Expected {network.Nt} entries, but received {conduit_length.size}')
        dist = conduit_length.reshape((-1, 1))

    weights = 1./dist
    weights = np.append(weights, -weights)
    grad = np.transpose(net.create_incidence_matrix(weights=weights, fmt='coo'))

    if Nc > 1:
        if include is None:
            include = range(Nc)
        num_included = len(include)

        # im = np.transpose(network.create_incidence_matrix(weights=weights, fmt='coo'))
        data = np.zeros((grad.data.size, num_included), dtype=float)
        rows = np.zeros((grad.data.size, num_included), dtype=float)
        cols = np.zeros((grad.data.size, num_included), dtype=float)

        pos = 0
        for n in include:
            rows[:, pos] = grad.row * Nc + n
            cols[:, pos] = grad.col * Nc + n
            data[:, pos] = grad.data
            pos += 1

        rows = np.ndarray.flatten(rows)
        cols = np.ndarray.flatten(cols)
        data = np.ndarray.flatten(data)
        mat_shape = (Nt * Nc, Np * Nc)
        grad = scipy.sparse.coo_matrix((data, (rows, cols)), shape=mat_shape)
    return scipy.sparse.csr_matrix(grad)


def delta(network,
          include: int | List[int] | None = None,
          exclude: int | List[int] | None = None,
          Nc: int | None = None) -> scipy.sparse.csr_matrix:
    r"""
    Constructs the matrix for differences (deltas)

    Parameters
    ----------
        network
            An instance with similar signatures as a MulticomponentTools or OpenPNM network object
            In the case of an OpenPNM network, the argument Nc has to be specified
        include: int|list[int]|None
            list of component IDs to include, all if set to None
        exclude: int|list[int]|None
            list of component IDs to exclude, no impact if include is set
        Nc: int|None
            number of components, needs to be specified if used if network doesn't have the function get_num_components
    Returns
    -------
        Delta matrix

    Notes
    -----
        Under the hood, this function creates a gradient matrix, where the
        'conduit lengths' are all set to 1.
    """

    if isinstance(network, ts.MulticomponentTools):
        net = network.get_network()
    else:
        net = network

    conns = net['throat.conns']
    weights = np.ones_like(conns[:, 0], dtype=float)
    weights = np.append(weights, -weights)

    return gradient(network=network, include=include, exclude=exclude, Nc=Nc, conduit_length=weights)
