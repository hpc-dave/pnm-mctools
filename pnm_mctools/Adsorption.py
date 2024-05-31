import numpy as np
from typing import Callable
try:
    from .NumericalDifferentiation import NumericalDifferentiation
except ImportError:
    from NumericalDifferentiation import NumericalDifferentiation


def AdsorptionSingleComponent(c,
                              dilute: int, adsorbed: int,
                              y_max, Vp, a_v, ftheta: Callable,
                              k_r: float = 1e6, type: str = 'Jacobian',
                              exclude=None):
    r"""
    Provides Jacobian and Defect for single component adsorption by means of a pseudo-reaction
    using numerical differentiation

    Parameters
    ----------
    c: array_like
        [Np, Nc] array of current component values
    dilute: int
        component ID of dilute species
    adsorbed: int
        component ID of adsorbed species
    y_max
        maximum surface load, usually in mol/m^2, can be array or single float
    Vp
        pore volume volume for scaling in m^3, ignored if set to 'None'
    a_v
        specific surface area in m^2/m^3
    ftheta: callable
        function object which computes the equilibrium relative amount of occupied surface sites for a given
        concentration of dilute and adsorbed species, signature of the function object needs to be:
        (c_0: array_like, c_1: array_like): array_like
        with c_0 as dilute species concentration, c_1 as adsorbed species concentration and returning the
        equilibrium value of the relative amount of occupied surface sites, the arrays are of size [Np, 1]
    k_r
        pseudo reaction rate constant
    type: str
        identifier, if Jacobian should be computed or simply the defect
    exclude
        component IDs which can be ommited during the numerical differentiation, either a single integer or
        a list of integers

    Notes
    -----
    This implementation treats adsorption as a super fast pseudo-reaction, with the difference between the
    equilibrium surface load and the current load as driving force. The reaction rate is defined by:
    .. math::

        r_{ad} = k_r * (\theta_{eq} - \theta)~\mathrm{in mol/(m^2 s)}

    The volumetric reaction rate is computed by:
    .. math::

        r_v = -a_v * r_{ad} ~\mathrm{in mol/(m^3 s)}

    Here a word of advice: The numerical differentiation is expensive, therefore it is not recommended to compute
    the Jacobian at each Newton-iteration. However, using only the Defect requires more iterations, so careful
    optimization is required by choosing between the two options!
    """
    A, b = None, None
    Np = c.shape[0]
    Nc = 1 if len(c.shape) == 1 else c.shape[1]
    V_pore = (np.ones((Np, 1), dtype=float) if Vp is None else Vp.reshape((-1, 1)))
    rows = np.arange(dilute, Np * Nc, Nc, dtype=int).reshape((-1))
    delta_n = adsorbed - dilute

    def Defect(c):
        c_f = c[:, dilute].reshape((-1, 1))
        c_ads = c[:, adsorbed].reshape((-1, 1))
        theta = c_ads / y_max
        theta_eq = ftheta(c_f, c_ads)
        r_ads = k_r * y_max * (theta_eq - theta)
        r_f = -a_v * r_ads
        G = np.zeros((Np * Nc, 1), dtype=float)
        G[rows, :] = r_f
        G[rows + delta_n, :] = r_ads
        return G

    if type == 'Defect':
        b = Defect(c)
    else:
        A, b = NumericalDifferentiation(c, defect_func=Defect, exclude=exclude)

    V_pore = np.hstack([V_pore for _ in range(Nc)]).reshape((-1, 1))
    b *= V_pore
    if A is not None:
        A = A.multiply(V_pore)

    return A, b
