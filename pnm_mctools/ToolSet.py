import numpy as np
import scipy
import scipy.sparse
import inspect
from . import NumericalDifferentiation


def GetLineInfo():
    r"""
    Provides information of calling point in the form: <path/to/file>: l. <line number> in <function name>
    """
    return f"{inspect.stack()[1][1]}: l.{inspect.stack()[1][2]} in {inspect.stack()[1][3]}"


def _apply_prescribed_bc(pore_labels, bc, num_components: int, n_c: int, A, x, b, type: str):
    r"""
    Enforces prescribed boundary conditions to the provided matrix and/or rhs vector

    Parameters
    ----------
    pore_labels: array_like
        pore ids of the affected pores
    bc: dict
        the values associated with the boundary condition, currently supported keywords
        are 'value' and 'prescribed'
    num_components: int
        number of implicitly coupled components in the linear equation system
    n_c: int
        affected component
    A: matrix
        matrix which may be manipulated
    x: numpy.ndarray
        solution vector
    b: numpy.ndarray
        rhs vector
    type: str
        type of algorithm/affected part of the LES, special treatments are required
        if the defect for Newton-Raphson iterations is computed

    Notes
    -----
    This function does have a specialization for CSR matrices, which is recommended for
    fast matrix-matrix and matrix-vector operations.
    """
    row_aff = pore_labels * num_components + n_c
    value = bc['prescribed'] if 'prescribed' in bc else bc['value']
    if b is not None:
        if type == 'Jacobian' or type == 'Defect':
            b[row_aff] = x[row_aff] - value
        else:
            b[row_aff] = value

    if (A is not None) and (type != 'Defect'):
        if scipy.sparse.isspmatrix_csr(A):
            # optimization for csr matrix (avoid changing the sparsity structure)
            # benefits are memory and speedwise (tested with 100000 affected rows)
            for r in row_aff:
                ptr = (A.indptr[r], A.indptr[r+1])
                A.data[ptr[0]:ptr[1]] = 0.
                pos = np.where(A.indices[ptr[0]: ptr[1]] == r)[0]
                A.data[ptr[0] + pos[0]] = 1.
        else:
            A[row_aff, :] = 0.
            A[row_aff, row_aff] = 1.
    return A, b


def _apply_rate_bc(pore_labels, bc, num_components: int, n_c: int, A, x, b, type: str):
    r"""
    Enforces rate boundary conditions to the provided matrix and/or rhs vector

    Parameters
    ----------
    pore_labels: array_like
        pore ids of the affected pores
    bc: dict
        the values associated with the boundary condition, currently supported keyword is 'rate'
    num_components: int
        number of implicitly coupled components in the linear equation system
    n_c: int
        affected component
    A: matrix
        matrix which may be manipulated
    x: numpy.ndarray
        solution vector
    b: numpy.ndarray
        rhs vector
    type: str
        type of algorithm/affected part of the LES, special treatments are required
        if the defect for Newton-Raphson iterations is computed

    Notes
    -----
    A rate is directly applied as explicit source term to pore and therefore ends
    up on the RHS of the LES.
    """
    if b is None:
        return A, b

    row_aff = pore_labels * num_components + n_c
    value = bc['rate']
    if isinstance(value, float) or isinstance(value, int):
        values = np.full(row_aff.shape, value/row_aff.size, dtype=float)
    else:
        values = value

    b[row_aff] -= values.reshape((-1, 1))

    return A, b


def _apply_outflow_bc(pore_labels, bc, num_components: int, n_c: int, A, x, b, type: str):
    r"""
    Enforces an outflow boundary conditions to the provided matrix and/or rhs vector

    Parameters
    ----------
    pore_labels: array_like
        pore ids of the affected pores
    bc: dict
        the values associated with the boundary condition, currently not in use here
    num_components: int
        number of implicitly coupled components in the linear equation system
    n_c: int
        affected component
    A: matrix
        matrix which may be manipulated
    x: numpy.ndarray
        solution vector
    b: numpy.ndarray
        rhs vector
    type: str
        type of algorithm/affected part of the LES, special treatments are required
        if the defect for Newton-Raphson iterations is computed

    Returns:
    Manipulated matrix A and rhs b. Currently, if more than one component is specified, the
    matrix will be converted into CSR format. Otherwise the output type depends on the input
    type. CSR will return a CSR matrix, all other types a LIL matrix.

    Notes
    -----
    An outflow pore is not integrated and not divergence free. The value in the affected
    pore is averaged from the connected pores, weighted by the respective fluxes.
    For convective contributions, the fluxes are independent of the outflow pore. In contrast,
    diffusive contributions require this averaged value to work properly.
    It is left to the user to make sure, that this is ALWAYS an outflow boundary, in the case
    of reverse flow the behavior is undefinend.
    This function does have a specialization for CSR matrices, which is recommended for
    fast matrix-matrix and matrix-vector operations.
    """
    row_aff = pore_labels * num_components + n_c

    if A is not None:
        if num_components > 1:
            A = scipy.sparse.csr_matrix(A)

        if scipy.sparse.isspmatrix_csr(A):
            # optimization for csr matrix (avoid changing the sparsity structure)
            # note that we expect here that the center value is allocated!
            # benefits are memory and speedwise (tested with 100000 affected rows)
            for r in row_aff:
                ptr = (A.indptr[r], A.indptr[r+1])
                ind = A.indices[ptr[0]: ptr[1]]
                mask = ind == r
                pos_nb = np.where(~mask)[0] + ptr[0]
                pos_c = np.where(mask)[0] + ptr[0]
                if num_components > 1:
                    pos_avg = [p for p in pos_nb if A.indices[p] % num_components == n_c]
                    pos_rem = [p for p in pos_nb if p not in pos_avg]
                    if pos_rem:
                        A.data[pos_rem] = 0.
                else:
                    pos_avg = pos_nb
                coeff = np.sum(A.data[pos_avg])
                coeff = -1. if coeff == 0 else coeff
                A.data[pos_c] = -coeff
        else:
            A = scipy.sparse.lil_matrix(A)
            coeff = np.sum(A[row_aff, :], axis=1) - A[row_aff, row_aff]
            A[row_aff, row_aff] = -coeff

    if b is not None:
        b[row_aff] = 0.

    return A, b


def ComputeRateForBC(bc, phi, rate_old=None):
    r"""
    Computes the values for a rate boundary, based on the the field values with
    the target to keep the all values at the boundary close to each other. As an example,
    to keep the pressure inside all affected pores at the same level.

    Notes
    -----
    Keep in mind, that the rate is defined as total rate with the unit [a.u.]/s, NOT as
    [a.u.]/(m²s)!
    """
    if bc is not None:
        # My brain was pretty cooked when writing this function, be sure to test it
        # before applying!
        raise ('Untested!')
    rate = bc['rate']
    if rate_old is None:
        if isinstance(rate, float) or isinstance(rate, int):
            rate_old = np.full(phi.shape, fill_value=rate/phi.size, dtype=float)
        else:
            rate_old = rate

    phi_avg = np.average(phi)/phi.size
    dphi = phi-phi_avg
    sum_phi = np.sum(np.abs(phi))
    sum_dphi = np.sum(np.abs(dphi))
    if sum_dphi == 0.:
        return rate_old
    rate_tot = np.sum(rate_old)
    sum_phi = sum_phi if sum_phi > 0. else 1.
    sum_phi = sum_phi if sum_phi > sum_dphi else sum_dphi
    drate_tot = sum_dphi / sum_phi * rate_tot
    rate_lower = dphi > 0.
    w_rate = np.zeros(dphi.shape, dtype=float)
    w_rate[rate_lower] = -drate_tot * dphi[rate_lower] / np.sum(dphi[rate_lower])
    w_rate[~rate_lower] = -drate_tot * dphi[~rate_lower] / np.sum(-dphi[~rate_lower])
    drate = w_rate * drate_tot
    if np.sum(drate) != 0.:
        raise ('determined changes of rate are not conservative')
    rate_new = rate_old + drate
    return rate_new


def ApplyBC(network, bc, A=None, x=None, b=None, type='Jacobian'):
    r"""
    Manipulates the provided Matrix and/or rhs vector according to the boundary conditions

    Parameters
    ----------
    network: any
        OpenPNM network with geometrical information
    bc: list/dict
        Boundary conditions in the form of a dictionary, where each label is associated with a certain type of BC
        In the case of multiple components, a list of dicts needs to be provided where the position in the list
        determines the component ID that the boundary condition is associated with
    A: matrix
        [Np, Np] Matrix to be manipulated, if 'None' is provided this will skipped
    x: array_like
        [Np, 1] numpy array with initial guess
    b: array_like
        [Np, 1] numpy array with rhs values
    type: str
        specifies the type of manipulation, especially for the prescribed value the enforcement differs between
        direct substitution and Newton-iterations

    Returns
    -------
    Return depends on the provided data, if A and b are not 'None', both are returned. Otherwise either A or b are returned.

    Notes
    -----
    Currently supported boundary conditions are:
        'noflow'     - not manipulation required
        'prescribed' - prescribes a value in the specified pore
        'value'      - alias for 'prescribed'
        'rate'       - adds a rate value to the pore
        'outflow'    - labels pore as outflow and interpolates value from connected pores

    The boundary conditions have to be provided by means of a list where each position in the list is associated with a
    component in the system. Each element in the list has to be a dictionary, where the key refers to the label of pores
    that the boundary condition is associated with. The value of the dictionary is again a dictionary with the type of
    boundary condition and the value.
    For single components, only the dictionary may be given as input parameter
    A code example:

    # The system of DGLs models 3 components and the network has two boundary regions 'inlet' and 'outlet'
    Nc = 3
    bc = [{}] * Nc
    bc[0]['inlet']  = {'prescribed': 1.}     # Component 0 has a prescribed value at the inlet with the value 1
    bc[0]['outlet'] = {'outflow'}            # at the outlet the species is allowed to leave the system (technically a set, any provided value will either way be ignored)
    bc[1]['inlet']  = {'rate': 0.1}          # Component 1 has an inflow rate with value 0.1
    bc[1]['outlet'] = {'outflow'}            # Component 1 is also allowed to leave the system
    bc[2]['inlet']  = {'noflow'}             # Component 2 is not allowed to enter of leave the system, technically this is not required to specify but the verbosity helps to address setup errors early on
    bc[2]['outlet'] = {'noflow'}             # Component 2 may also not leave at the outlet, e.g. because it's adsorbed to the surface

    """
    if len(bc) == 0:
        print(f'{GetLineInfo()}: No boundary conditions were provided, consider removing function altogether!')

    if A is None and b is None:
        raise ('Neither matrix nor rhs were provided')
    if type == 'Jacobian' and A is None:
        raise (f'No matrix was provided although {type} was provided as type')
    if type == 'Jacobian' and b is not None and x is None:
        raise (f'No initial values were provided although {type} was specified and rhs is not None')
    if type == 'Defect' and b is None:
        raise (f'No rhs was provided although {type} was provided as type')

    num_pores = network.Np
    num_rows = A.shape[0] if A is not None else b.shape[0]
    num_components = int(num_rows/num_pores)
    if (num_rows % num_pores) != 0:
        raise (f'the number of matrix rows now not consistent with the number of pores,\
               mod returned {num_rows % num_pores}')
    if b is not None and num_rows != b.shape[0]:
        raise ('Dimension of rhs and matrix inconsistent!')

    if isinstance(bc, dict) and isinstance(list(bc.keys())[0], int):
        bc = list(bc)
    elif not isinstance(bc, list):
        bc = [bc]

    for n_c, boundary in enumerate(bc):
        for label, param in boundary.items():
            bc_pores = network.pores(label)
            if 'noflow' in param:
                continue  # dummy so we can write fully specified systems
            elif 'prescribed' in param or 'value' in param:
                A, b = _apply_prescribed_bc(pore_labels=bc_pores,
                                            bc=param,
                                            num_components=num_components, n_c=n_c,
                                            A=A, x=x, b=b,
                                            type=type)
            elif 'rate' in param:
                A, b = _apply_rate_bc(pore_labels=bc_pores,
                                      bc=param,
                                      num_components=num_components, n_c=n_c,
                                      A=A, x=x, b=b,
                                      type=type)
            elif 'outflow' in param:
                A, b = _apply_outflow_bc(pore_labels=bc_pores,
                                         bc=param,
                                         num_components=num_components, n_c=n_c,
                                         A=A, x=x, b=b,
                                         type=type)
            else:
                raise (f'unknown bc type: {param.keys()}')

    if A is not None:
        A.eliminate_zeros()

    if A is not None and b is not None:
        return A, b
    elif A is not None:
        return A
    else:
        return b


class MulticomponentTools:
    r"""
    Object with convenient methods for developing multicomponent transport models with OpenPNM pore networks.

    Parameters
    ----------
    network: OpenPNM network
        network with the geometrical information
    num_components: int
        number of components that shall be modelled with this toolset
    bc: list
        list of boundary conditions for each component. May be 'None', then no
        boundary conditions will be applied

    Notes
    -----

    A multicomponent model is expected to adhere to the conservation law:
    \[f
        \frac{\partial}{\partial t}\phi + \nabla \cdot J - S = 0
    \]
    with the extensive variable $\phi$, time $t$, flux $J$ and source $S$. This toolset provides convenient
    functions to compute gradients, divergences, time derivatives and fluxes. In the case of multiple coupled
    components, the matrix is organized block-wise so the rows of coupled components computed at a single node
    (or pore respectively) are located adjacent to each other.
    Note, that here the term 'divergence' is used, although contemporary literature employs sums and references
    Kirchhoff's law. For simplicity, this notation is dropped and only 'divergence' used.
    """
    def __init__(self, network, num_components: int = 1, bc=None):
        self.network = network
        self.bc = bc
        self.num_components = num_components
        self._ddt = {}
        self._grad = {}
        self._div = {}
        self._upwind = {}

    def _get_include(self, include, exclude):
        r"""
        computes the include list from a list of excluded components

        Parameters
        ----------
        include: list
            list of included components
        exclude: list
            list of components to exclude

        Returns
        -------
        list of included parameters
        """
        if include is None and exclude is None:
            return include

        if include is not None:
            return include
        else:
            return [n for n in range(self.num_components) if n not in exclude]

    def _construct_ddt(self, dt: float, weight='pore.volume', include=None, exclude=None):
        r"""
        Computes the discretized matrix for the partial time derivative

        Parameters
        ----------
        network:
            OpenPNM network
        dt: float
            discretized timestep
        num_components: int
            number of coupled components components
        weight:
            Weight to used for discretization, by default the pore volume is used

        Returns
        -------
        [Np, Np] sparse CSR matrix with transient terms

        Notes
        -----
        By default, a finite volume discretization is assumed, therefore the standard form of
        the partial derivative is given by

        \iiint \frac{\partial}{\partial t} \mathrm{d}V \approx \frac{\Delta V}{\Delta t}

        Note that here the integrated variable is ommitted in the description, as it will be provided
        either by the solution vector for implicit treatment and by the field for explicit components
        """

        include = self._get_include(include, exclude)

        network = self.network
        num_components = self.num_components

        if dt <= 0.:
            raise (f'timestep is invalid, following constraints were violated: {dt} !> 0')
        if num_components < 1:
            raise (f'number of components has to be positive, following value was provided: {num_components}')

        dVdt = network[weight].copy() if isinstance(weight, str) else weight
        dVdt /= dt
        if isinstance(dVdt, float) or isinstance(dVdt, int):
            dVdt = np.full((network.Np, 1), fill_value=dVdt, dtype=float)
        dVdt = dVdt.reshape((-1, 1))
        if num_components > 1:
            if dVdt.size == network.Np:
                dVdt = np.tile(A=dVdt, reps=num_components)
            if include is not None:
                mask = np.asarray([n in include for n in range(num_components)], dtype=bool).reshape((1, -1))
                mask = np.tile(A=mask, reps=(dVdt.shape[0], 1))
                dVdt[~mask] = 0.
        ddt = scipy.sparse.spdiags(data=[dVdt.ravel()], diags=[0])
        return ddt

    def _construct_grad(self, include=None):
        """
        Constructs the gradient matrix

        Args:
            self: network with geometric information
            include (list): list of component IDs to include, all if set to None

        Returns:
            Gradient matrix

        Notes:
            The direction of the gradient is given by the connections specified in the network,
            mores specifically from conns[:, 0] to conns[:, 1]
        """
        network = self.network
        num_components = self.num_components

        conns = network['throat.conns']
        p_coord = network['pore.coords']
        dist = np.sqrt(np.sum((p_coord[conns[:, 0], :] - p_coord[conns[:, 1], :])**2, axis=1))
        weights = 1./dist
        weights = np.append(weights, -weights)
        if num_components == 1:
            return np.transpose(network.create_incidence_matrix(weights=weights, fmt='csr'))
        else:
            if include is None:
                include = range(num_components)
            num_included = len(include)

            im = np.transpose(network.create_incidence_matrix(weights=weights, fmt='coo'))
            data = np.zeros((im.data.size, num_included), dtype=float)
            rows = np.zeros((im.data.size, num_included), dtype=float)
            cols = np.zeros((im.data.size, num_included), dtype=float)

            pos = 0
            for n in include:
                rows[:, pos] = im.row * self.num_components + n
                cols[:, pos] = im.col * self.num_components + n
                data[:, pos] = im.data
                pos += 1

            rows = np.ndarray.flatten(rows)
            cols = np.ndarray.flatten(cols)
            data = np.ndarray.flatten(data)
            mat_shape = (self.network.Nt * self.num_components, self.network.Np * self.num_components)
            grad = scipy.sparse.coo_matrix((data, (rows, cols)), shape=mat_shape)
            return scipy.sparse.csr_matrix(grad)

    def _compute_flux_matrix(self, *args):
        r"""
        computes matrix of size [Np*Nc, Nt*Nc], where all arguments are multiplied with the last argument

        Parameters
        ----------
        Factors to multiply with the final argument, where the final argument is a [Nt*Nc, Np*Nc] matrix

        Returns
        -------
        [Np*Nc, Nt*Nc] sized matrix
        """
        network = self.network
        Nc = self.num_components
        fluxes = args[-1].copy()
        for i in range(len(args)-1):
            arg = args[i]
            if isinstance(arg, list) and len(arg) == Nc:
                fluxes = fluxes.multiply(np.tile(np.asarray(arg), network.Nt))
            elif isinstance(arg, np.ndarray):
                _arg = np.tile(arg.reshape(-1, 1), reps=(1, Nc)) if arg.size == network.Nt else arg
                fluxes = fluxes.multiply(_arg.reshape(-1, 1))
            else:
                fluxes = fluxes.multiply(args[i])
        return fluxes

    def _construct_div(self, weights=None, custom_weights: bool = False, include=None):
        """
        Constructs divergence matrix

        Parameters
        ----------
        weights: any
            weights for each connection, if 'None' all values will be set to 1
        custom_weights: bool
            identifier, if the weights are customized and shall not be manipulated, extended
            or in any other way made fit with the expected size
        include: list
            identifer, which components should be included in the divergence, all other
            rows will be set to 0

        Returns:
            Divergence matrix
        """
        network = self.network
        num_components = self.num_components
        _weights = None
        if custom_weights:
            if weights is None:
                raise ('custom weights were specified, but none were provided')
            _weights = np.flatten(weights)
            if _weights.shape[0] < network.Nt*num_components*2:
                _weights = np.append(-_weights, _weights)
        else:
            _weights = np.ones(shape=(network.Nt)) if weights is None else np.ndarray.flatten(weights)
            if _weights.shape[0] == network.Nt:
                _weights = np.append(-_weights, _weights)

        if num_components == 1:
            div_mat = network.create_incidence_matrix(weights=_weights, fmt='coo')
        else:
            if include is None:
                include = range(num_components)
            num_included = len(include)

            ones = np.ones(shape=(network.Nt*2))
            div_mat = network.create_incidence_matrix(weights=ones, fmt='coo')
            data = np.zeros((div_mat.data.size, num_included), dtype=float)
            rows = np.zeros((div_mat.data.size, num_included), dtype=float)
            cols = np.zeros((div_mat.data.size, num_included), dtype=float)
            pos = 0
            for n in include:
                rows[:, pos] = div_mat.row * num_components + n
                cols[:, pos] = div_mat.col * num_components + n
                if custom_weights:
                    beg, end = n * network.Nt * 2, (n + 1) * network.Nt * 2 - 1
                    data[:, pos] = _weights[beg: end]
                else:
                    data[:, pos] = _weights
                pos += 1
            rows = np.ndarray.flatten(rows)
            cols = np.ndarray.flatten(cols)
            data = np.ndarray.flatten(data)
            mat_shape = (network.Np * num_components, network.Nt * num_components)
            div_mat = scipy.sparse.coo_matrix((data, (rows, cols)), shape=mat_shape)

        # converting to CSR format for improved computation
        div_mat = scipy.sparse.csr_matrix(div_mat)

        def div(*args):
            fluxes = self.Fluxes(*args)
            return div_mat * fluxes

        return div

    def _construct_upwind(self, fluxes, include=None):
        r"""
        Constructs a [Nt, Np] matrix representing a directed network based on the upwind
        fluxes

        Parameters
        ----------
        fluxes : any
            fluxes which determine the upwind direction, see below for more details
        include : list
            a list of integers to specify for which components the matrix should be constructed,
            for components which are not listed here, the rows will be 0. If 'None' is provided,
            all components will be selected

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
        num_components = self.num_components
        network = self.network
        if num_components == 1:
            # check input
            if isinstance(fluxes, float) or isinstance(fluxes, int):
                _fluxes = np.zeros((network.Nt)) + fluxes
            elif fluxes.size == network.Nt:
                _fluxes = fluxes
            else:
                raise ('invalid flux dimensions')
            weights = np.append((_fluxes < 0).astype(float), _fluxes > 0)
            return np.transpose(network.create_incidence_matrix(weights=weights, fmt='csr'))
        else:
            if include is None:
                include = range(num_components)
            num_included = len(include)

            im = np.transpose(network.create_incidence_matrix(fmt='coo'))

            data = np.zeros((im.data.size, num_included), dtype=float)
            rows = np.zeros((im.data.size, num_included), dtype=int)
            cols = np.zeros((im.data.size, num_included), dtype=int)

            pos = 0
            for n in include:
                rows[:, pos] = im.row * num_components + n
                cols[:, pos] = im.col * num_components + n
                data[:, pos] = im.data
                pos += 1

            if isinstance(fluxes, float) or isinstance(fluxes, int):
                # single provided value
                _fluxes = np.zeros((network.Nt)) + fluxes
                weights = np.append(_fluxes < 0, _fluxes > 0)
                pos = 0
                for n in include:
                    data[:, pos] = weights
                    pos += 1
            elif (isinstance(fluxes, list) and len(fluxes) == num_components)\
                    or (isinstance(fluxes, np.ndarray) and fluxes.size == num_components):
                # a list of values for each component
                _fluxes = np.zeros((network.Nt))
                pos = 0
                for n in include:
                    _fluxes[:] = fluxes[n]
                    weights = np.append(_fluxes < 0, _fluxes > 0)
                    data[:, pos] = weights
                    pos += 1
            elif fluxes.size == network.Nt:
                # fluxes for each throat, e.g. for single component or same convective fluxes
                # for each component
                weights = np.append(fluxes < 0, fluxes > 0)
                pos = 0
                for n in include:
                    data[:, pos] = weights.reshape((network.Nt*2))
                    pos += 1
            elif (len(fluxes.shape)) == 2\
                and (fluxes.shape[0] == network.Nt)\
                    and (fluxes.shape[1] == num_components):
                # each throat has different fluxes for each component
                pos = 0
                for n in include:
                    weights = np.append(fluxes[:, n] < 0, fluxes[:, n] > 0)
                    data[:, pos] = weights.reshape((network.Nt*2))
                    pos += 1
            else:
                raise ('fluxes have incompatible dimension')

            rows = np.ndarray.flatten(rows)
            cols = np.ndarray.flatten(cols)
            data = np.ndarray.flatten(data)
            mat_shape = (network.Nt * num_components, network.Np * num_components)
            upwind = scipy.sparse.coo_matrix((data, (rows, cols)), shape=mat_shape)
            return scipy.sparse.csr_matrix(upwind)

    def _convert_include_to_key(self, include):
        r"""
        provides a key based on the include list

        Parameters
        ----------
        include: list
            list of integers, representing component ids to include for further treatment
            if 'None', all components will be included
        Returns
        -------
        A tuple of the sorted list, which can be used as key in dictionaries
        """
        return tuple(range(self.num_components) if include is None else sorted(include))

    def DDT(self, dt: float = 1., weight='pore.volume', include=None):
        r"""
        Computes partial time derivative matrix

        Parameters
        ----------
        dt: float
            discretized time step size
        weight: any
            a weight which can be applied to the time derivative, usually that should be
            the volume of the computational cell
        include:
            an ID or list of IDs which should be included in the matrix, if 'None' is provided,'
            all values will be used

        Returns
        -------
        Matrix in CSR format
        """
        include = [include] if isinstance(include, int) else include
        key = self._convert_include_to_key(include)
        if key not in self._ddt:
            self._ddt[key] = self._construct_ddt(dt=dt, weight=weight, include=include)
        return self._ddt[key]

    def Gradient(self, include=None):
        r"""
        Computes a gradient matrix

        Parameters
        ----------
        include:
            int or list of ints with IDs to include, if 'None' is provided
            all IDs will be included

        Returns
        -------
        a gradient matrix in CSR-format
        """
        include = [include] if isinstance(include, int) else include
        key = self._convert_include_to_key(include)
        if key not in self._grad:
            self._grad[key] = self._construct_grad(include=include)
        return self._grad[key]

    def Fluxes(self, *args):
        r"""
        computes fluxes from a set of arguments

        Parameters
        ----------
        args:
            Set of arguments, where the last argument is either a gradient matrix,
            vector of fluxes at the throats or flux matrix. All arguments before will
            be multiplied with this value
        """
        return self._compute_flux_matrix(*args)

    def Divergence(self, weights=None, custom_weights: bool = False, include=None):
        include = [include] if isinstance(include, int) else include
        key = self._convert_include_to_key(include)
        if key not in self._div:
            self._div[key] = self._construct_div(weights=weights, custom_weights=custom_weights)
        return self._div[key]

    def Upwind(self, fluxes, include=None):
        include = [include] if isinstance(include, int) else include
        key = self._convert_include_to_key(include)
        if key not in self._upwind:
            self._upwind[key] = self._construct_upwind(fluxes=fluxes, include=include)
        return self._upwind[key]

    def ApplyBC(self, A=None, x=None, b=None, type='Jacobian'):
        return ApplyBC(self.network, bc=self.bc, A=A, x=x, b=b, type=type)

    def NumericalDifferenciation(self, c, defect_func, dc: float = 1e-6, mem_opt: str = 'full', type: str = 'Jacobian'):
        if type == 'Jacobian':
            return NumericalDifferentiation(c=c, defect_func=defect_func, dc=dc, type=mem_opt)
        elif type == 'Defect':
            return defect_func(c).reshape((-1, 1))
        else:
            raise ('Unknown type for numerical differentiation')
