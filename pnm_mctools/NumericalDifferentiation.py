import numpy as np
import scipy
import time
from inspect import signature


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

    dc = np.full((x_0.size, 1), fill_value=dc_value)
    x_0_abs = np.abs(x_0)
    mask = (x_0_abs < dc_value) & (x_0_abs > 0.)
    dc[mask] *= np.abs(x_0[mask])
    return dc


def _apply_numerical_differentiation_lowmem(c, defect_func, dc: float = 1e-6, exclude=None):
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
    if len(c.shape) > 2:
        raise ('Input array has invalid dimension, only 1D or 2D arrays are allowed!')
    exclude = [] if exclude is None else exclude
    exclude = [exclude] if isinstance(exclude, int) else exclude
    if not isinstance(exclude, list):
        raise TypeError('the provided exclude list is not an integer or a list of integers')

    num_param = len(signature(defect_func)._parameters)
    single_param = num_param == 1
    if num_param == 0:
        raise ValueError('The provided defect function does not take any arguments!')
    elif num_param > 2:
        raise ValueError('Number of arguments for defect function is larger than 2!')

    if single_param:
        G_0 = defect_func(c).reshape((-1, 1))
    else:
        G_0 = defect_func(c, None).reshape((-1, 1))

    num_cols = c.size
    J_col = [None] * num_cols
    x_0 = c.reshape(-1, 1)
    dc = _compute_dc(x_0, dc)

    Nc = 1 if len(c.shape) == 1 else c.shape[1]
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

    J = scipy.sparse.csr_matrix(scipy.sparse.hstack(J_col))
    return J, G_0


def _apply_numerical_differentiation_full(c, defect_func, dc: float = 1e-6, exclude=None):
    r"""
    Conducts numerical differentiation with focus on simplicity

    Parameters
    ----------
    c: array_like
        array with scalar values, which serve as input for the defect function
    defect_func: func
        function which computes the defect with signature array_like(array_like, int),
        where the second argument refers to the manipulated row
    dc: float
        base value for differentiation interval
    exclude
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
    if len(c.shape) > 2:
        raise ('Input array has invalid dimension, only 1D or 2D arrays are allowed!')
    exclude = [] if exclude is None else exclude
    exclude = [exclude] if isinstance(exclude, int) else exclude
    if not isinstance(exclude, list):
        raise TypeError('the provided exclude list is not an integer or a list of integers')

    G_0 = defect_func(c).reshape((c.size, 1))
    num_cols = c.size
    try:
        J = np.zeros((c.size, c.size), dtype=float)
    except MemoryError:
        print('Numerical differentiation with the full matrix exceeds locally available memory, invoking low memory variant instead!')
        return _apply_numerical_differentiation_lowmem(c=c, defect_func=defect_func, dc=dc)

    x_0 = c.reshape(-1, 1)
    dc = _compute_dc(x_0, dc)

    num_param = len(signature(defect_func)._parameters)
    single_param = num_param == 1
    if num_param == 0:
        raise ValueError('The provided defect function does not take any arguments!')
    elif num_param > 2:
        raise ValueError('Number of arguments for defect function is larger than 2!')

    Nc = 1 if len(c.shape) == 1 else c.shape[1]
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
    J = scipy.sparse.csr_matrix(J)
    return J, G_0


def NumericalDifferentiation(c, defect_func, dc: float = 1e-6, type: str = 'full', exclude=None):
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
        and sparse columns respectively.
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
    if type == 'full':
        return _apply_numerical_differentiation_full(c=c, defect_func=defect_func, dc=dc, exclude=exclude)
    elif type == 'low_mem':
        return _apply_numerical_differentiation_lowmem(c=c, defect_func=defect_func, dc=dc, exclude=exclude)
    else:
        raise (f'Unknown type: {type}')


def _testing():
    sizes = [50, 100, 500, 1000, 2000, 5000]

    for size in sizes:
        print(f'size -> {size}')
        c = np.zeros((size, 1), dtype=float)

        J_0 = np.arange(1., c.size+1, dtype=float)
        J_0 = np.tile(J_0, reps=[c.size, 1])
        J_0 += (np.arange(c.size) * c.size).reshape((-1, 1))
        J_0 = np.matrix(J_0)

        def Defect(c, *args):
            return J_0 * c.reshape((c.size, 1))

        tic = time.perf_counter_ns()
        J, G = NumericalDifferentiation(c, defect_func=Defect, type='full')
        toc = time.perf_counter_ns()
        err = np.max(np.abs((J-J_0)/J_0))
        print(f'block: {(toc-tic)*1e-9:1.2e} s - max error: {err}')
        tic = time.perf_counter_ns()
        J, G = NumericalDifferentiation(c, defect_func=Defect, type='low_mem')
        toc = time.perf_counter_ns()
        err = np.max(np.abs((J-J_0)/J_0))
        print(f'low mem: {(toc-tic)*1e-9:1.2e} s - max error: {err}')

    print('finished')
