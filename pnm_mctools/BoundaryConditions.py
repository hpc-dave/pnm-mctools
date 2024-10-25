from typing import Tuple, Any, Set
import scipy
import numpy as np
from . import ToolSet as ts


def unpack_info(network, bc) -> Tuple[Any, int, int, int, Any]:
    if isinstance(network, ts.MulticomponentTools):
        net = network.get_network()
        if bc is None:
            bc = network.get_bc()
    else:
        net = network
        if bc is None:
            raise ValueError('boundary conditions need to be specified, if the network cannot provide them')
    Np = net.num_pores()
    Nt = net.Nt
    return net, Np, Nt, bc


def set(mt: ts.MulticomponentTools, **kwargs):
    id, label, bc = 0, 'None', None
    Nc = mt.get_num_components()

    for key, value in kwargs.items():
        match key:
            case 'id':
                if not isinstance(value, int):
                    raise ValueError(f'The provided ID ({value}) is not an integer value!')
                if value < 0:
                    raise ValueError(f'The provided ID ({value}) has to be positive!')
                elif value > Nc:
                    raise ValueError(f'The provided ID ({value}) exceeds the number of components ({Nc})! Do you take into account 0-indexing?')  # noqa: E501
                id = value
            case 'label':
                if not isinstance(value, str):
                    raise TypeError(f'The provided label ({value}) needs to be a string!')
                label = value
            case 'bc':
                if isinstance(value, float) or isinstance(value, int):
                    bc = {'prescribed': value}
                elif isinstance(value, dict):
                    bc = value
                elif isinstance(value, Set):
                    for e in value:
                        if isinstance(e, str) and e == 'outflow':
                            bc = {'outflow': None}
                else:
                    raise ValueError(f'The provided BC ({bc}) cannot be converted to a standard bc')
            case _:
                raise ValueError(f'Unknown key value provided: {key}')

    if label == 'None':
        raise ValueError('No label was provided for the BC! Cannot continue')
    if bc is None:
        raise ValueError('The provided BC is None! Cannot continue')

    mt.get_bc()[id][label] = bc


def apply_prescribed(pore_labels, bc, num_components: int, n_c: int, A, x, b, type: str):
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


def apply_rate(pore_labels, bc, num_components: int, n_c: int, A, x, b, type: str):
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


def apply_outflow(pore_labels, bc, num_components: int, n_c: int, A, x, b, type: str):
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
                if coeff == 0.:
                    # this scenario can happen under specific scenarios, e.g. a
                    # diffusion coefficient of the value 0 or purely convective flow
                    # with an upwind scheme. Then we need to set the correlation of
                    # the components appropriately
                    coeff = -1.
                    A.data[pos_nb] = coeff
                    A.data[pos_c] = -pos_nb.size * coeff
                else:
                    A.data[pos_c] = -coeff
        else:
            A = scipy.sparse.lil_matrix(A)
            coeff = np.sum(A[row_aff, :], axis=1) - A[row_aff, row_aff]
            A[row_aff, row_aff] = -coeff

    if b is not None:
        b[row_aff] = 0.

    return A, b


def apply(network, bc=None, A=None, x=None, b=None, type='Jacobian'):
    r"""
    Manipulates the provided Matrix and/or rhs vector according to the boundary conditions

    Parameters
    ----------
    network: any
        OpenPNM network with geometrical information or instance of MulticomponentTools
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
        'noflow'     - no manipulation required
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
    bc[0]['outlet'] = {'outflow'}            # at the outlet the species is allowed to leave the system (technically a set, any provided value will either way be ignored)  # noqa: E501
    bc[1]['inlet']  = {'rate': 0.1}          # Component 1 has an inflow rate with value 0.1
    bc[1]['outlet'] = {'outflow'}            # Component 1 is also allowed to leave the system
    bc[2]['inlet']  = {'noflow'}             # Component 2 is not allowed to enter of leave the system, technically this is not required to specify but the verbosity helps to address setup errors early on  # noqa: E501
    bc[2]['outlet'] = {'noflow'}             # Component 2 may also not leave at the outlet, e.g. because it's adsorbed to the surface  # noqa: E501
    """

    if A is None and b is None:
        raise ValueError('Neither matrix nor rhs were provided')
    if type == 'Jacobian' and A is None:
        raise ValueError(f'No matrix was provided although {type} was provided as type')
    if type == 'Jacobian' and b is not None and x is None:
        raise ValueError(f'No initial values were provided although {type} was specified and rhs is not None')
    if type == 'Defect' and b is None:
        raise ValueError(f'No rhs was provided although {type} was provided as type')

    net, Np, _, bc = unpack_info(network=network, bc=bc)
    num_rows = A.shape[0] if A is not None else b.shape[0]
    num_components = int(num_rows/Np)

    if (num_rows % Np) != 0:
        raise ValueError(f'the number of matrix rows now not consistent with the number of pores,\
               mod returned {num_rows % Np}')
    if b is not None and num_rows != b.shape[0]:
        raise ValueError('Dimension of rhs and matrix inconsistent!')

    if isinstance(bc, dict) and isinstance(list(bc.keys())[0], int):
        bc = list(bc)
    elif not isinstance(bc, list):
        bc = [bc]

    for n_c, boundary in enumerate(bc):
        for label, param in boundary.items():
            bc_pores = net.pores(label)
            if 'noflow' in param:
                continue  # dummy so we can write fully specified systems
            elif 'prescribed' in param or 'value' in param:
                A, b = apply_prescribed(pore_labels=bc_pores,
                                        bc=param,
                                        num_components=num_components, n_c=n_c,
                                        A=A, x=x, b=b,
                                        type=type)
            elif 'rate' in param:
                A, b = apply_rate(pore_labels=bc_pores,
                                  bc=param,
                                  num_components=num_components, n_c=n_c,
                                  A=A, x=x, b=b,
                                  type=type)
            elif 'outflow' in param:
                A, b = apply_outflow(pore_labels=bc_pores,
                                     bc=param,
                                     num_components=num_components, n_c=n_c,
                                     A=A, x=x, b=b,
                                     type=type)
            else:
                raise ValueError(f'unknown bc type: {param.keys()}')

    if A is not None:
        A.eliminate_zeros()

    if A is not None and b is not None:
        return A, b
    elif A is not None:
        return A
    else:
        return b
