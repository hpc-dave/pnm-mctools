from typing import Callable, List, Any, Tuple
import numpy as np
import scipy
try:
    from . import NumericalDifferentiation as num_diff
except ImportError:
    import NumericalDifferentiation as num_diff


def _extract_and_sort_parameters(c: np.ndarray, network, Vp, a_v) -> Tuple[Any, int, int, np.ndarray, np.ndarray]:
    r"""
    Helper function to determine common parameters used in adsorption models from input values

    Parameters
    ----------
    c: np.ndarray
        array with variables in the form [Np, Nc]
    network
        an object similar to a dict, an OpenPNM network or a MulticomponentTools class
    Vp
        pore volume array, string identifier or None
    a_v
        specific surface area

    Returns
    -------
    A tuple with following parameters: network, number of pores, number of components,
    pore volumes, specific surface area
    """
    net = network.get_network() if hasattr(network, 'get_network') else network
    Np = c.shape[0]
    Nc = 1 if len(c.shape) == 1 else c.shape[1]
    if Vp is None:
        V_pore = np.ones((Np, 1), dtype=float)
    else:
        V_pore = Vp if isinstance(Vp, np.ndarray) else net[Vp]

    if a_v is None:
        spec_surf = np.ones((Np, 1), dtype=float)
    else:
        spec_surf = a_v if isinstance(a_v, np.ndarray) else net[a_v]

    return net, Np, Nc, V_pore.reshape((Np, -1)), spec_surf.reshape((Np, -1))


def Linear(c_f, K):
    r"""
    Linear adsorption, e.g. in the limit case of highly dilute species

    Parameters
    ----------
    c_f
        concentration in the fluid phase, usually in mol/m^3
    K
        equilibrium constant, usually in m^3/(m^2 mol)

    Returns
    -------
    surface load
    """
    return c_f * K


def Langmuir(c_f, K, y_max):
    r""""
    Single component Langmuir adsorption model

    Parameters
    ----------
    c_f
        concentration in the fluid phase, usually in mol/m^3
    K
        equilibrium constant, usually in m^3/mol
    y_max
        maximum surface load, usually in m^2/mol

    Returns
    -------
    surface coverage
    """
    y = y_max * c_f*K/(1+c_f*K)
    if isinstance(y, np.ndarray) and y.size != c_f.size:
        raise ValueError('Apparently the dimensions are inconsistent and yield a broken field')
    return y


def Freundlich(c_f, K, n):
    r"""
    Freundlich isotherm

    Parameters
    ----------
    c_f
        concentration in the fluid phase, usually in mol/m^3
    K
        equilibrium constant, usually in m^(3n)/(mol^n m^2)
    n
        exponent of concentration

    Returns
    -------
    surface load
    """
    y = K * c_f**(1./n)
    if isinstance(y, np.ndarray) and y.size != c_f.size:
        raise ValueError('Apparently the dimensions are inconsistent and yield a broken field')
    return y


def AdsorptionSingleComponent(c,
                              dilute: int, adsorbed: int,
                              Vp, a_v, y_func: Callable,
                              k_r: float = 1e6, type: str = 'Jacobian',
                              exclude=None, dc=1e-6):
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
    Vp
        pore volume volume for scaling in m^3, ignored if set to 'None'
    a_v
        specific surface area in m^2/m^3
    ftheta: callable
        function object which computes the surface load for a given concentration of dilute and adsorbed species,
        signature of the function object needs to be:
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

    For simplicity, the partial surface coverage was chosen, however for practical application this can be treated
    as surface load, e.g. for Linear or Freundlich isotherms which do not exhibit a limit. The final surface load is
    determined by the maximum surface load y_max by
    .. math::

        Y = y_{max} * \theta

    So by appropriate manipulation of y_max, e.g. by y_max = 1, the surface load is directly computed by the isotherm

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
        y_ads = c[:, adsorbed].reshape((-1, 1))
        y_ads_eq = y_func(c_f, y_ads)
        r_ads = k_r * (y_ads_eq - y_ads)
        r_f = -a_v.reshape((-1, 1)) * r_ads
        G = np.zeros((Np * Nc, 1), dtype=float)
        G[rows, :] = r_f
        G[rows + delta_n, :] = r_ads
        return G

    if type == 'Defect':
        b = Defect(c)
    else:
        A, b = num_diff.conduct_numerical_differentiation(c, defect_func=Defect, exclude=exclude, dc=dc,
                                                          type='constrained')

    V_pore = np.hstack([V_pore for _ in range(Nc)]).reshape((-1, 1))
    b *= V_pore
    if A is not None:
        A = A.multiply(V_pore)

    return A, b


def single_linear(c, c_old,
                  K_func: Callable,
                  dt: float,
                  component_id: int | List[int] | None = None,
                  Vp: np.ndarray | str | None = 'pore.volume',
                  a_v: np.ndarray | str | None = 'pore.specific_surface_area',
                  network=None,
                  stype: str = 'Jacobian'):
    r"""
    Provides Jacobian with only main diagonal entries and Defect for adsorption

    Parameters
    ----------
    c: array_like
        [Np, Nc] array of current bulk values
    c_old: array_like
        [Np, Nc] array of previous bulk values
    K_func: callable
        function object which computes the adsorption constant K for a given concentration of dilute species,
        signature of the function object needs to be:
            (c_0: array_like): array_like
        with c_0 as species concentration in the bulk and returning the
        equilibrium value of adsorbed species, the arrays are of size [Np, 1]
    component_id: int| List[int] | None
        component ID of  dilute species, by default all species will be included
    Vp: np.ndarray | str | None
        pore volume volume for scaling (usually in m^3),
        if a string is provided it will query the value from the network
    a_v: np.ndarray | str| None
        specific surface area (usually in m^2/m^3),
        if a string is provided it will query the value from the network
    network
        An object similar to a dictionary, an OpenPNM network or a MulticomponentTools class
    type: str
        identifier, if Jacobian should be computed or simply the defect

    Notes
    -----
    This implementation provides a fast alternative for computing the Jacobian of a linearized adsorption.
    The underlying adsorption is assumed to look as follows:

    .. math::
        \phi_{ads} = K(c) * \phi_{bulk} * a_v

    with the adsorption constant K and the specific surface area a_v

    In discretized form this becomes:

    .. math::
        \int \frac{\partial \phi_{ads}}{\partial t} \mathrm{d}V
        \approx a_v * \Delta V * \frac{(K(c^{n+1})*\phi^{n+1})-(K(c^{n})*\phi^{n})}{\Delta t}

    Subsequently the defect is defined as:
    .. math::
        G = a_v * \Delta V * \frac{(K(c^{n+1})*\phi^{n+1})-(K(c^{n})*\phi^{n})}{\Delta t}

    And the main diagonal entries of the Jacobian:
    .. math::
        \alpha_0 = a_v * \Delta V * \frac{K(c^{n+1})*\phi^{n+1}}{\Delta t}
    """
    stype = stype.lower()
    if stype not in ['jacobian', 'defect', 'direct']:
        raise ValueError(f'Unknown type for the computation: {stype} - allowed: {["jacobian", "defect", "direct"]}')

    A, b = None, None
    _, Np, Nc, V_pore, sp_surf = _extract_and_sort_parameters(c=c, network=network, Vp=Vp, a_v=a_v)

    if component_id is None:
        c_id = range(Nc)
    elif isinstance(component_id, int):
        c_id = [component_id]

    # here, the last dimension is used as a convenient way to store current and old step
    # meaning the [:,:,0] is used for the current and [:,:,1] for the old step
    b = np.zeros((Np, Nc, 2), dtype=float)

    # The right hand side or defect are computed by
    # K_new * phi_new / dt * a_v * Vp, -K_old * phi_old / dt * a_v * Vp
    # so later we can use both components or simply add them up to get the defect
    b[:, c_id, 0] = K_func(c[:, c_id]).reshape((Np, -1)) * c[:, c_id]
    b[:, c_id, 1] = -K_func(c_old[:, c_id]).reshape((Np, -1)) * c_old[:, c_id]
    b[:, c_id, :] = np.multiply(b[:, c_id, :], np.expand_dims(sp_surf * V_pore / dt, axis=2))

    if (stype == 'jacobian') or (stype == 'direct'):
        A = scipy.sparse.spdiags(data=[b[:, :, 0].ravel()], diags=[0], format='csr')

    if (stype == 'jacobian') or (stype == 'defect'):
        # for the Jacobian we compute the defect by adding both components up
        b = np.sum(b, axis=2).reshape((-1, 1))
    else:
        # for the direct solving, only the explict components need to be taken to
        # the right hand side
        # note, that special care has to be given, if this is supposed to be the
        # a LINEARIZED, partially implicit formulation, then the rhs is missing
        # the deviating order components!
        b = b[:, :, 1].reshape((-1, 1))
        b *= -1.

    if A is not None:
        return A, b
    else:
        return b


def multi_component(c, c_old,
                    theta_func: Callable,
                    dt: float,
                    component_id: int | List[int] | None = None,
                    Vp: np.ndarray | None = None,
                    a_v: np.ndarray | None = None,
                    network=None,
                    stype: str = 'Jacobian',
                    diff_type: str = 'constrained'):
    r"""
    Provides Jacobian and Defect for adsorption via numerical differentiation

    Parameters
    ----------
    c: array_like
        [Np, Nc] array of current bulk values
    c_old: array_like
        [Np, Nc] array of previous bulk values
    theta_func: callable
        function object which computes the surface load for a given concentration of dilute species,
        signature of the function object needs to be:
            (c_0: array_like): array_like
        with c_0 as species concentration in the bulk and returning the
        equilibrium value of adsorbed species, the arrays are of size [Np, 1]
    component_id: int| List[int] | None
        component ID of  dilute species, by default all species will be included
    Vp: np.ndarray | str | None
        pore volume volume for scaling (usually in m^3),
        if a string is provided it will query the value from the network
    a_v: np.ndarray | str| None
        specific surface area (usually in m^2/m^3),
        if a string is provided it will query the value from the network
    network
        An object similar to a dictionary, an OpenPNM network or a MulticomponentTools class
    stype: str
        identifier, if Jacobian should be computed or simply the defect
    diff_type: str
        identifier for optimization of the numerical differentiation, by default the
        defect is considered to be constrained within a single pore
    Notes
    -----
    Here, the Jacobian and Defect fo implementing adsorption of multiple components are computed
    The underlying adsorption is assumed to look as follows:

    .. math::
        \phi_{ads} = \theta(c) * a_v

    with the surface load \theta and the specific surface area a_v

    In discretized form this becomes:

    .. math::
        \int \frac{\partial c_{ads}}{\partial t} \mathrm{d}V
        \approx a_v * \Delta V * \frac{K(c^{n+1})-K(c^{n})}{\Delta t}

    Subsequently the defect is defined as:
    .. math::
        G = a_v * \Delta V * \frac{K(c^{n+1})-K(c^{n})}{\Delta t}

    With that, the Jacobian can be computed by numerical differentiation.
    """
    stype = stype.lower()
    if stype not in ['jacobian', 'defect']:
        raise ValueError(f'Unknown type for the computation: {stype} - allowed types: {["jacobian", "defect"]}')

    A, b = None, None
    _, _, Nc, V_pore, sp_surf = _extract_and_sort_parameters(c=c, network=network, Vp=Vp, a_v=a_v)

    alpha = np.multiply(V_pore, sp_surf/dt)

    if component_id is None:
        c_id = range(Nc)
    elif isinstance(component_id, int):
        c_id = [component_id]

    theta_old = theta_func(c_old[:, c_id])

    def Defect(c):
        G = np.zeros_like(c)
        G[:, c_id] = theta_func(c[:, c_id]) - theta_old
        G = np.multiply(G, alpha)
        return G.reshape((-1, 1))

    if stype == 'jacobian':
        A, b = num_diff.conduct_numerical_differentiation(c, defect_func=Defect, type=diff_type)
    else:
        b = Defect(c)

    if A is not None:
        return A, b
    else:
        return b
