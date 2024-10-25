import numpy as np
from ToolSet import _compute_flux_matrix, MulticomponentTools
import Operators as ops


def compute_rates(mc: MulticomponentTools, *args):
    r"""
    computes transport rates from a set of arguments

    Parameters
    ----------
        mc: MulticomponentTools
            instance of the multicomponent tools object
        args:
            Set of arguments, where the last argument is either a gradient matrix,
            vector of fluxes at the throats or flux matrix. All arguments before will
            be multiplied with this value
    """
    return _compute_flux_matrix(mc.get_network().Nt, mc.get_num_components(), *args)


def compute_fluxes(mc: MulticomponentTools, *args):
    r"""
    computes fluxes from a set of arguments

    Parameters
    ----------
        args:
            Set of arguments, where the last argument is either a gradient matrix,
            vector of fluxes at the throats or flux matrix. All arguments before will
            be multiplied with this value

    Notes
    -----
        alias for compute_rates
    """
    return compute_rates(mc, *args)


def compute_pore_residence_time(Q: np.ndarray,
                                network=None,
                                approach: str = 'min',
                                A_dir=None,
                                Vp: np.ndarray | str = 'pore.volume'):
    r"""
    Computes the residence time for each pore

    Parameters
    ----------
        Q: np.ndarray
            array of size [Nt] of the flow rates in the throats
        approach: str
            type of approach for determining the pore residence time, current options are:
            - 'inflow' sum of flow rates into the pore
            - 'outflow' sum of flow rates out of the pore
            - 'min' minimum of the two above mentioned approaches
        A_dir: sparse matrix|None
            a sparse matrix of size [Np, Nt] with direction of the flow rates, the values should be either [-1, 0, 1]!
            by default the appropriate matrix will be determined from the underlying network
        Vp: np.ndarray|str
            array of size [Np] of the individual pore volumes or a string to infer those values
            from the underlying network

    Returns
    -------
        array of size [Np] with the residence time per pore

    Notes
    -----
        The residence time is computed by
        \[f
            \tau_i = \frac{V_{p, i}}{Q_{total},i}
        \]
        where the flow rate through a pore can be based on the in- or the outflow
        \[f
            Q_{total,i} = Q_{in/out,i} = \sum_j Q_{in/out, ij}
        \]
        or the minimum of the two options:
        \[f
            Q_{total,i} = min(\sum_j Q_{in, ij}, \sum_j Q_{out, ij})
        \]
    """

    if A_dir is None:
        if network is None:
            raise ValueError('neither `A_dir` nor `network` are defined, cannot continue')
        A_dir = ops.sum(network=network, Nc=1)

    if isinstance(Vp, str):
        if isinstance(network, MulticomponentTools):
            Vp = network.get_network()[Vp]
        else:
            Vp = network[Vp]

    approach_options = ['inflow', 'outflow', 'min']
    if approach not in approach_options:
        raise ValueError(f'Cannot compute residence time, unknown approach: {approach}! Available options are {approach_options}')  # noqa: E501

    Q_dir = A_dir.multiply(Q.reshape((-1)))
    tau_in = np.zeros_like(Vp)  # dummy for inflow residence time
    tau_out = tau_in.copy()     # dummy for outflow residence time
    if (approach == 'inflow') or (approach == 'min'):
        Q_in = Q_dir.multiply(Q_dir < 0.)  # matrix solely with negative flow rates
        Q_in = np.abs(np.sum(Q_in, axis=1))  # sum up all the inflow rates
        tau_in = Vp.reshape((-1, 1))/Q_in  # determine the local residence time
    if (approach == 'outflow') or (approach == 'min'):
        Q_out = Q_dir.multiply(Q_dir > 0.)  # matrix solely with positive flow rates
        Q_out = np.abs(np.sum(Q_out, axis=1))  # sum up all the outflow rates
        tau_out = Vp.reshape((-1, 1))/Q_out  # determine the local residence time

    tau = np.min(np.hstack((tau_in, tau_out)), axis=1)  # maximum or the residence times
    return np.array(tau)
