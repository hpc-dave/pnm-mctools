import math
import numpy as np


def calibrated_conductance(conn, pore_radii, throat_radii, throat_density, throat_viscosity, rate,
                           gamma: float, C_0: float, E_0: float, F: float, m: float, n: float):
    r"""
    Computes the hydraulic conductance according to Eghbalmanesh and Fathiganjehlou (10.1016/j.ces.2023.119396)

    Parameters
    ----------
    conn: array_like
        connectivity of each throat, stored in an [Nt, 2] array
    pore_radii: array_like
        array with pore radii of size [Np,] in m
    throat_radii: array_like
        array with throat radii of size [Nt,] in m
    throat_density: array_like
        array fluid density at the throats of size [Nt,] in kg/m^3
    throat_viscosity: array_like
        array fluid viscosity at the throats of size [Nt,] in Pa s
    rate: array_like
        array of volumetric flow rates at each pore of size [Nt,] in m^3/s
    gamma: float
        flow pattern constant
    C_0: float
        laminar contraction coefficient
    E_0: float
        laminar expansion coefficient
    F: float
        global radius scaling factor
    m: float
        expansion exponent
    n: float
        contraction exponent

    Returns
    -------
    array of size [Nt,] with conductances in m^3/(Pa s)
    """
    r_ij = throat_radii.reshape((-1, 1)) * F
    mu = throat_viscosity.reshape((-1, 1))
    rho = throat_density.reshape((-1, 1))
    rate = np.zeros(r_ij.shape, dtype=float) if rate is None else rate
    _rate = np.abs(rate)
    _rate[_rate < 1e-15] = 1e-15    # safety to avoid propagating NaNs
    r_4 = r_ij**4
    r_i = pore_radii[conn[:, 0]].reshape((-1, 1))
    r_j = pore_radii[conn[:, 1]].reshape((-1, 1))
    Re_ij = 2 * rho * _rate/(math.pi * mu * r_ij)
    A_ij = 8 * mu / (math.pi * r_4)
    C_ij = rho/(2 * math.pi**2 * r_4) * _rate * ((C_0/Re_ij)**n + 1./(2**n) * (1 - r_ij/r_i)**n)
    E_ij = rho/(2 * math.pi**2 * r_4) * _rate * ((E_0/Re_ij)**m + (1 - r_ij/r_j)**n)
    G_ij = gamma * rho * _rate / (2 * math.pi**2) * (1/r_i**4 - 1/r_j**4)

    # noflow = _rate < 1e-15
    # C_ij[noflow] = 0.
    # E_ij[noflow] = 0.

    return (1./(A_ij + C_ij + E_ij - G_ij)).reshape((-1, 1))


def GetConductanceObject(network, F, m, n, C_0: float = 26, E_0: float = 27, gamma: float = 1):

    def _compute_cond(throat_density, throat_viscosity, rate_prev=None):
        return calibrated_conductance(conn=network['throat.conns'],
                                      pore_radii=network['pore.diameter'] * 0.5,
                                      throat_radii=network['throat.diameter'] * 0.5,
                                      throat_density=throat_density,
                                      throat_viscosity=throat_viscosity,
                                      rate=rate_prev,
                                      gamma=gamma,
                                      C_0=C_0, E_0=E_0, F=F, m=m, n=n)
    return _compute_cond


def GetCalibratedConductanceModel(F, m, n, C_0: float = 26, E_0: float = 27, gamma: float = 1):

    def model_func(target: any, conn: str, pore_diameter: str, throat_diameter: str,
                   throat_density: str, throat_viscosity: str, rate: str):
        return calibrated_conductance(conn=target[conn],
                                      pore_radii=target[pore_diameter]*0.5,
                                      throat_radii=target[throat_diameter]*0.5,
                                      throat_density=target[throat_density],
                                      throat_viscosity=target[throat_viscosity],
                                      rate=target[rate],
                                      gamma=gamma,
                                      C_0=C_0,
                                      E_0=E_0,
                                      F=F,
                                      m=m,
                                      n=n)

    model = {
        'model': model_func,
        'conn': 'throat.conn',
        'pore_diameter': 'pore.diameter',
        'throat_diameter': 'throat.diameter',
        'throat_density': 'throat.density',
        'throat_viscosity': 'throat.viscosity',
        'flux': 'throat.hydraulic_rate'
    }

    return model
