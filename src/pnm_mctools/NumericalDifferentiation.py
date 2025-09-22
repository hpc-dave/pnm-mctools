import numpy as np
import scipy
import time
from inspect import signature
from typing import Callable
# from pathos.multiprocessing import ProcessingPool as Pool
# from pathos.helpers import cpu_count


def _compute_dc(x_0, dc_value: float):
    r"""
    computes array with discrete interval value for numerical differentiation

    Parameters
    ----------
    x_0:
        initial value vector
    dc_value: float
        discrete interval value

    Returns
    -------
    1-dim vector with interval values for each row of the LES
    """
    dc = np.abs(x_0) * dc_value
    dc[dc < dc_value] = dc_value
    return dc


def _apply_numerical_differentiation_lowmem(c, defect_func,
                                            dc: np.ndarray,
                                            exclude=None,
                                            axis: int = None,
                                            is_locally_constrained: bool = False):
    r"""
    Conducts numerical differentiation with focus on low memory demand

    Parameters
    ----------
    c: array_like
        array with scalar values, which serve as input for the defect function
    defect_func: func
        function which computes the defect with signature array_like(array_like, float)
    dc: float
        base value for differentiation interval
    exclude
        component IDs for which the numerical differentiation shall not be conducted

    Returns
    -------
    tuple of numerically determined Jacobian and defect

    Notes
    -----
    To save memory during the computation, the matrix entries are stored per column
    in a sparse array and later stacked to form the full sparse array.
    The in between conversion leads to a slight overhead compared with the approach
    to directly add the components into an existing array, but decreases memory
    demand significantly, especially for large matrices (>5000 rows)
    """

    single_param = len(signature(defect_func)._parameters) == 1

    if single_param:
        G_0 = defect_func(c).reshape((-1, 1))
    else:
        G_0 = defect_func(c, None).reshape((-1, 1))

    x_0 = c.reshape(-1, 1)

    Nc = c.shape[1]
    num_cols = c.size
    J_col = [None] * num_cols
    for col in range(num_cols):
        if (col % Nc) not in exclude:
            x = x_0.copy()
            x[col] += dc[col]
            if single_param:
                G_loc = defect_func(x.reshape(c.shape)).reshape((-1, 1))
            else:
                G_loc = defect_func(x.reshape(c.shape), col).reshape((-1, 1))
            # With a brief profiling, coo_arrays seem to perform best as
            # sparse storage format during assembly, need to investigate further
            J_col[col] = scipy.sparse.coo_array((G_loc-G_0)/dc[col])
        else:
            J_col[col] = scipy.sparse.coo_array(np.zeros_like(G_0))

    J = scipy.sparse.hstack(J_col)
    return J, G_0


def _apply_numerical_differentiation_full(c: np.ndarray,
                                          defect_func: Callable,
                                          dc: np.ndarray,
                                          exclude: list[int]):
    r"""
    Conducts numerical differentiation with focus on simplicity

    Parameters
    ----------
    c: array_like
        array with scalar values, which serve as input for the defect function
    defect_func: Callable
        function which computes the defect with signature array_like(array_like, int),
        where the second argument refers to the manipulated row
    dc: float
        base value for differentiation interval
    exclude: int | list[int]
        component IDs for which the numerical differentiation shall not be conducted

    Returns
    -------
    tuple of numerically determined Jacobian and defect

    Notes
    -----
    Here, a dense array is initialized and all entries places there during the computation
    including zero-values. This is currently the speedwise best performing variant and
    allows for comparatively simple debugging. However, it also does lead to a forbiddingly
    high memory demand for large systems (> 2000 rows)
    """

    num_cols = c.size
    try:
        J = np.zeros((c.size, c.size), dtype=float)
    except MemoryError:
        print('Numerical differentiation with the full matrix exceeds locally available memory, invoking low memory variant instead!')     # noqa: E501
        return _apply_numerical_differentiation_lowmem(c=c, defect_func=defect_func, dc=dc)

    single_param = len(signature(defect_func)._parameters) == 1
    x_0 = c.reshape(-1, 1)
    if single_param:
        G_0 = defect_func(c).reshape((-1, 1))
    else:
        G_0 = defect_func(c, None).reshape((-1, 1))

    Nc = c.shape[1]
    for col in range(num_cols):
        if (col % Nc) in exclude:
            continue
        x = x_0.copy()
        x[col] += dc[col]
        if single_param:
            G_loc = defect_func(x.reshape(c.shape)).reshape((-1, 1))
        else:
            G_loc = defect_func(x.reshape(c.shape), col).reshape((-1, 1))
        J[:, col] = ((G_loc-G_0)/dc[col]).reshape((-1))

    return J, G_0


def _apply_numerical_differentiation_locally_constrained(c: np.ndarray,
                                                         defect_func: Callable,
                                                         dc: np.ndarray,
                                                         exclude: int | list[int] = None):
    r"""
    Conducts numerical differentiation, optimized for locally constrained defects, i.e. reaction

    Parameters
    ----------
    c: array_like
        array with scalar values, which serve as input for the defect function
    defect_func: Callable
        function which computes the defect with signature array_like(array_like, int),
        where the second argument refers to the manipulated row
        **IMPORTANT** The defect MUST NOT have a dependency along the first axis!
    dc: float
        base value for differentiation interval
    exclude: int | list[int]
        component IDs for which the numerical differentiation shall not be conducted

    Returns
    -------
    tuple of numerically determined Jacobian and defect

    Notes
    -----
    Here, only the Jacobian with the local 'blocks' are computed.
    **IMPORTANT** The defect MUST NOT have a dependency along the first axis!
    """
    Nc = c.shape[1]
    single_param = len(signature(defect_func)._parameters) == 1

    if exclude is None:
        exclude = []

    # we assume that each 'cell' is responsible for Nc coupled components
    # therefore leading to a block-wise structure of the Jacobian. That
    # means, that each row requires Nc values. So we extend store the
    # row ids, col ids and values of interest in an (N_cell, Nc, Nc)-size array and later on
    # reshape it to a 1D vector
    shape_extended = list(c.shape)
    shape_extended.append(Nc)

    row_col_id = np.arange(0, c.size, Nc)       # temporary variable to set the row and column ids
    rows = np.zeros(shape_extended, dtype=int)  # store associated row values
    cols = np.zeros_like(rows, dtype=int)       # store associated column values
    values = np.zeros_like(cols, dtype=float)   # store the numerical differences
    dc = dc.reshape(c.shape)                    # reshape the differences for later use
    if single_param:
        G_0 = defect_func(c).reshape((-1, Nc))
    else:
        G_0 = defect_func(c, None).reshape((-1, Nc))

    # Here's the crucial optimization:
    # We loop through the COMPONENTS, instead of each column since each block
    # is independent of each other
    for n_c in range(Nc):
        row_col_id_l = (row_col_id + n_c).reshape((-1, 1))
        rows[:, n_c, :] = row_col_id_l
        cols[:, :, n_c] = row_col_id_l
        if n_c in exclude:
            continue
        c_perturb = c.copy()
        c_perturb[:, n_c] += dc[:, n_c]
        if single_param:
            G_loc = defect_func(c_perturb).reshape((-1, Nc))
        else:
            G_loc = defect_func(c_perturb, row_col_id_l).reshape((-1, Nc))
        values[:, :, n_c] = (G_loc-G_0) / dc[:, n_c].reshape((-1, 1))

    # to accomodate that we want to exclude certain values, even if they are dependent
    # on the specified excluded component (e.g. because we are lazy and didn't take the
    # explicit component out of the defect function), the row based values need to be
    # removed/set to 0
    if len(exclude) > 0:
        values[:, exclude, :] = 0.
    values = values.reshape((-1))
    rows = rows.reshape((-1))
    cols = cols.reshape((-1))
    J = scipy.sparse.coo_matrix((values, (rows, cols)))
    J = scipy.sparse.csr_matrix(J)
    return J, G_0


def _apply_numerical_differentiation_exploit_sparsity(
        network,
        c: np.ndarray,
        defect_func: Callable,
        dc: float,
        exclude: int | list[int] | None = None,
        stencil_size: int = 1,
        dtype=float,
        opt: dict | None = None,
        parallelism: int = 1):
    r"""
    Conducts numerical differentiation, exploiting the sparsity structure of the network
    Parameters
    ----------
    network: OpenPNM Network object
        The network object is used to determine the sparsity structure of the Jacobian.
    c: array_like
        array with scalar values, which serve as input for the defect function
    defect_func: Callable
        function which computes the defect with signature array_like(array_like)
    dc: float
        base value for differentiation interval
    exclude: int | list[int]
        component IDs for which the numerical differentiation shall not be conducted
    stencil_size: int
        number of additional adacent pores to include in the sparsity structure
    dtype: data-type
        desired data-type of the scalars in the Jacobian
    opt: dict | None
        an optimization parameter, which will provide some reusable data

    Returns
    -------
    tuple of numerically determined Jacobian and defect

    Notes
    -----
    Here, we exploit the sparsity structure of the network to speed up the numerical
    differentiation. The basic assumption is that the influence of a perturbance in
    one pore is limited to its direct neighbours and potentially some additional
    pores, depending on the stencil size. This way, we can determine the influence
    of multiple perturbances in one run of the defect function, significantly reducing
    the computational cost.
    """
    tic = time.perf_counter_ns()
    num_pores = network.num_pores()
    Nc = c.shape[1]
    if exclude is None:
        exclude = []
    elif isinstance(exclude, int):
        exclude = [exclude]

    if not isinstance(opt, dict) or 'independent_pores' not in opt:
        # analyse the sparsity structure
        adj = network.create_adjacency_matrix(weights=np.ones_like(network['throat.conns'], dtype=bool), fmt='csr')
        pore_independent = []
        list_pores_rem = np.arange(0, num_pores)
        while len(list_pores_rem) > 0:
            list_p_independent = list_pores_rem.copy()
            i = 0
            while i < len(list_p_independent):
                p_loc = list_p_independent[i]
                stencil_loc = set(adj.indices[adj.indptr[p_loc]:adj.indptr[p_loc+1]])
                for _ in range(stencil_size):
                    for adj_p in stencil_loc.copy():
                        stencil_loc.update(adj.indices[adj.indptr[adj_p]:adj.indptr[adj_p+1]])
                list_p_independent = [p for p in list_p_independent if p not in stencil_loc or p == p_loc]
                i += 1
            pore_independent.append(np.array(list_p_independent))
            list_pores_rem = [p for p in list_pores_rem if p not in list_p_independent]
        if isinstance(opt, dict):
            opt['independent_pores'] = pore_independent
            opt['adjacency_matrix'] = adj
    else:
        pore_independent = opt['independent_pores']
        adj = opt['adjacency_matrix']

    t_prep = time.perf_counter_ns() - tic
    # prepare basic variables
    if isinstance(dc, np.ndarray):
        dc_arr = dc
    else:
        dc_arr = _compute_dc(c.reshape((-1, 1)), dc)         # perturbance values
    G0 = defect_func(c).reshape((-1, 1))                 # reference defect
    adj += scipy.sparse.eye(adj.shape[0], dtype=bool)    # to include self-dependency
    shape_jac = (num_pores*Nc, num_pores*Nc)             # shape of the Jacobian
    J = scipy.sparse.coo_matrix(shape_jac, dtype=dtype)  # initialize empty sparse matrix

    # conduct the numerical differentiation in block of independent pores
    # here we basically loop through each column of the Jacobian and exploit
    # that the influence of perturbances is limited to some columns, indicated
    # by the pore connectivity. This way, we can determine the influence of multiple
    # perturbances in one run of the defect function, significantly reducing
    # the computational cost
    tic = time.perf_counter_ns()

    # below is an implementation for a potential parallel optimization.
    # however, the defect function does not update. Once the core issue is
    # resolved, this could lead to speedup for large systems
    # if parallelism > 1:
    #     parallelism = min(parallelism, cpu_count())
    # pool = Pool(parallelism)

    # def inner_loop(i: int):
    #     p_loc = pore_independent[i]
    #     J_loc = scipy.sparse.coo_matrix(shape_jac, dtype=dtype)  # initialize empty sparse matrix
    #     p_aff = np.hstack(tuple(adj.indices[adj.indptr[p]:adj.indptr[p+1]] for p in p_loc))
    #     num_conn = np.array(np.sum(adj[p_loc] != 0, axis=1)).ravel()
    #     for n in range(Nc):
    #         if n in exclude:
    #             continue
    #         cols = p_loc * Nc + n
    #         rows = p_aff * Nc + n
    #         c_loc = c.reshape(-1, 1).copy()
    #         c_loc[cols] += dc_arr[cols]
    #         G_loc = defect_func(c_loc.reshape(c.shape)).reshape((-1, 1))
    #         values = np.array((G_loc-G0)/dc).ravel()  # avoid potential type issues if a matrix is returned
    #         values = values[rows]
    #         cols = np.hstack([np.asarray(np.tile([cols[i]], reps=(num_conn[i]))).reshape(-1) for i in range(len(cols))]) # noqa: E501
    #         J_loc += scipy.sparse.coo_matrix((values, (rows, cols)), shape=shape_jac, dtype=dtype)
    #     return J_loc

    # result = pool.map(inner_loop, range(len(pore_independent)))
    # for J_loc in result:
    #     J += J_loc
    for p_loc in pore_independent:
        p_aff = np.hstack(tuple(adj.indices[adj.indptr[p]:adj.indptr[p+1]] for p in p_loc))
        num_conn = np.array(np.sum(adj[p_loc] != 0, axis=1)).ravel()
        for n in range(Nc):
            if n in exclude:
                continue
            cols = p_loc * Nc + n
            rows = p_aff * Nc + n
            c_loc = c.reshape(-1, 1).copy()
            c_loc[cols] += dc_arr[cols]
            G_loc = defect_func(c_loc.reshape(c.shape)).reshape((-1, 1))
            values = np.array((G_loc-G0)/dc).ravel()  # avoid potential type issues if a matrix is returned
            values = values[rows]
            cols = np.hstack([np.asarray(np.tile([cols[i]], reps=(num_conn[i]))).reshape(-1) for i in range(len(cols))])
            J += scipy.sparse.coo_matrix((values, (rows, cols)), shape=shape_jac, dtype=dtype)
    J = scipy.sparse.csr_matrix(J)
    t_diff = time.perf_counter_ns() - tic
    print(f'prep: {t_prep*1e-9} s - diff: {t_diff*1e-9} s')
    return J, G0


def conduct_numerical_differentiation(c: np.ndarray, defect_func: Callable, dc: float = 1e-6, type: str = 'full',
                                      exclude: int | list[int] | None = None, axis: int = None,
                                      network=None, stencil_size: int = 1, opt: dict | None = None):
    r"""
    Conducts numerical differentiation

    Parameters
    ----------
    c: array_like
        array with scalar values, which serve as input for the defect function
    defect_func: func
        function which computes the defect with signature array_like(array_like, int)
        where the second argument refers to the manipulated row
    dc: float
        base value for differentiation-interval
    type: str
        specifier for optimization of the process, currently supported arguments are
        'full' and 'low_mem', for the allocation of an intermediated dense matrix
        and sparse columns respectively. The option 'constrained' can be applied, if
        the changes in defect are constrained to each pore, e.g. in the case of reaction
    exclude: int|list[int]|None
        component IDs for which the numerical differentiation shall not be conducted
    axis: int
        alternative to the `type` label for consistency with the mrm package
    network: OpenPNM Network object
        If provided, the sparsity structure of the network is exploited to speed up
        the numerical differentiation. This option is only available for `type='full'`.
        The network object is used to determine the sparsity
        structure of the Jacobian.
    stencil_size: int
        number of additional adacent pores to include in the sparsity structure

    Returns
    -------
    tuple of numerically determined Jacobian and defect

    Notes
    -----
    To save memory during the computation, the matrix entries are stored per column
    in a sparse array and later stacked to form the full sparse array.
    The in between conversion leads to a slight overhead compared with the approach
    to directly add the components into an existing array, but decreases memory
    demand significantly, especially for large matrices (>5000 rows)
    """
    if len(c.shape) > 2:
        raise ('Input array has invalid dimension, only 2D arrays are allowed!')
    elif len(c.shape) < 2:
        raise ('Input array has to have a second dimension, indicating the number of components')
    type_l = type
    if axis is not None:
        if axis == 0 and type is None:
            type_l = 'full'
        elif axis == 1:
            type_l = 'constrained'
        else:
            raise ValueError('axis value as to be either 0 or 1!')

    exclude = [] if exclude is None else exclude
    exclude = [exclude] if isinstance(exclude, int) else exclude
    if not isinstance(exclude, list):
        raise TypeError('the provided exclude list is not an integer or a list of integers')

    num_param = len(signature(defect_func)._parameters)
    if num_param == 0:
        raise ValueError('The provided defect function does not take any arguments!')
    elif num_param > 2:
        raise ValueError('Number of arguments for defect function is larger than 2!')

    dc_arr = _compute_dc(c.reshape((-1, 1)), dc)

    if type_l == 'constrained':
        J, G_0 = _apply_numerical_differentiation_locally_constrained(c=c,
                                                                      defect_func=defect_func,
                                                                      dc=dc_arr,
                                                                      exclude=exclude)
    elif type_l == 'full':
        if network is not None:
            J, G_0 = _apply_numerical_differentiation_exploit_sparsity(network=network,
                                                                       c=c,
                                                                       defect_func=defect_func,
                                                                       dc=dc_arr,
                                                                       stencil_size=1,
                                                                       exclude=exclude,
                                                                       opt=opt)
        else:
            J, G_0 = _apply_numerical_differentiation_full(c=c,
                                                           defect_func=defect_func,
                                                           dc=dc_arr,
                                                           exclude=exclude)
    elif type_l == 'low_mem':
        J, G_0 = _apply_numerical_differentiation_lowmem(c=c,
                                                         defect_func=defect_func,
                                                         dc=dc_arr,
                                                         exclude=exclude)
    else:
        raise (f'Unknown type: {type}')
    return scipy.sparse.csr_matrix(J), G_0.reshape((-1, 1))


if __name__ == '__main__':  # pragma: no cover
    import openpnm as op

    # test sparsity exploiting version
    shapes = [[5, 5, 5], [10, 10, 10], [20, 20, 20], [30, 30, 30]]

    for shape in shapes:
        Nc = 3
        print(f'shape -> {shape} - Nc -> {Nc}: size -> {np.prod(shape)*Nc}')
        c = np.ones((np.prod(shape), Nc), dtype=float)
        pn = op.network.Cubic(shape=shape, spacing=1)
        J_0 = pn.create_adjacency_matrix(weights=pn['throat.conns']+1, fmt='coo')
        rows = np.hstack([J_0.row*Nc + n for n in range(Nc)])
        cols = np.hstack([J_0.col*Nc + n for n in range(Nc)])
        data = np.hstack([J_0.data * J_0.size * (n+1) for n in range(Nc)])
        J_0 = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(c.size, c.size), dtype=float)
        J_0 += scipy.sparse.spdiags([np.arange(1, J_0.shape[0]+1)], [0], format='csr')

        def Defect(c):
            return J_0 * c.reshape((-1, 1))

        opt = {}
        tic = time.perf_counter_ns()
        J, G = _apply_numerical_differentiation_exploit_sparsity(network=pn, c=c, defect_func=Defect, dc=1e-6, opt=opt)
        toc = time.perf_counter_ns()
        if J.nnz == J_0.nnz and np.all(J.indices == J_0.indices) and np.all(J.indptr == J_0.indptr):  # noqa: E501
            err = np.max((J.data/J_0.data)-1)
        else:
            err = np.inf
        print(f'runtime (init): {(toc-tic)*1e-9:1.2e} s - max error: {err}')

        tic = time.perf_counter_ns()
        J, G = _apply_numerical_differentiation_exploit_sparsity(network=pn, c=c, defect_func=Defect, dc=1e-6, opt=opt)
        toc = time.perf_counter_ns()
        if J.nnz == J_0.nnz and np.all(J.indices == J_0.indices) and np.all(J.indptr == J_0.indptr):  # noqa: E501
            err = np.max((J.data/J_0.data)-1)
        else:
            err = np.inf
        print(f'runtime (opt): {(toc-tic)*1e-9:1.2e} s - max error: {err}')

    sizes = [50, 100, 500, 1000, 2000, 5000]

    for size in sizes:
        print(f'size -> {size}')
        # c = np.ones((size, 3), dtype=float)
        c = np.ones((size, 1), dtype=float)

        J_0 = np.arange(1., c.size+1, dtype=float)
        J_0 = np.tile(J_0, reps=[c.size, 1])
        J_0 += (np.arange(c.size) * c.size).reshape((-1, 1))
        J_0 = np.matrix(J_0)

        def Defect(c, *args):
            # f = c[:, 0] * c[:, 1] - 0.5*c[:, 2]
            # g = np.zeros_like(c)
            # g[:, 0] = f
            # g[:, 1] = 2.* f
            # g[:, 2] = -f
            # return g
            # return np.arange(0., c.size).reshape(c.shape) * c  # for debugging
            return J_0 * c.reshape((c.size, 1))

        tic = time.perf_counter_ns()
        J, G = conduct_numerical_differentiation(c, defect_func=Defect, type='full')
        toc = time.perf_counter_ns()
        err = np.max(np.abs((J-J_0)/J_0))
        print(f'block: {(toc-tic)*1e-9:1.2e} s - max error: {err}')

        tic = time.perf_counter_ns()
        J, G = conduct_numerical_differentiation(c, defect_func=Defect, type='low_mem')
        toc = time.perf_counter_ns()
        err = np.max(np.abs((J-J_0)/J_0))
        print(f'low mem: {(toc-tic)*1e-9:1.2e} s - max error: {err}')

        tic = time.perf_counter_ns()
        J, G = conduct_numerical_differentiation(c, defect_func=Defect, type='constrained')
        toc = time.perf_counter_ns()
        print(f'constrained: {(toc-tic)*1e-9:1.2e} s')

    print('finished')
