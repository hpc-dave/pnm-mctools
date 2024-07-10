import math
import numpy as np

def _default_mapping(type:str) -> list:
    r"""
    Provides a default mapping list for the calibration

    Parameters
    ----------
    type: str
        type of original extraction method, currently supported are:
            maximum_ball
            snow

    Returns
    -------
    list with mapping parameter of the form: [ label_target: str, label_source: str, factor_multiplicator: float]
    """
    map_keys = []
    if type.lower() == 'maximum_ball':
        map_keys.append(['throat.calibration_radius', 'throat.radius', 1.])
        map_keys.append(['pore.calibration_radius', 'pore.radius', 1.])
        map_keys.append(['throat.calibration_length', 'throat.conduit_lengths.throat', 1.])
    elif type.lower() == 'snow':
        map_keys.append(['throat.calibration_radius', 'throat.inscribed_diameter', 0.5])
        map_keys.append(['pore.calibration_radius', 'pore.inscribed_diameter', 0.5])
        map_keys.append(['throat.calibration_length', 'throat.total_length', 1.])
    else:
        raise ValueError('Unknown type: ' + type)
    return map_keys


def PrepareCalibratedValues(network, type: str = None, map_keys: list = None) -> None:
    r"""
    prepares the dedicated value arrays for the use with the calibrated conductance

    Parameters
    ----------
    network: OpenPNM.network | dict
        an OpenPNM network or dictionary holding the keys which need to be mapped
    type: str
        optional parameter for specifying the type of extraction method used to generate the
        network, by default it will try to determine automatically if maximum ball or snow were
        used
    map_keys: list
        optional parameter to specify custom mappings for the calibration, where each entry in the
        list has to be of the form: [ label_target: str, label_source: str, factor_multiplicator: float]
    """
    if (type is None) and (map_keys is None):
        if 'pore.clay_volume' in network:
            type = 'maximum_ball'
        elif 'pore.inscribed_diameter' in network and 'pore.equivalent_diameter' in network:
            type = 'snow'
        else:
            raise ValueError('Cannot determine network type automatically')

    if map_keys is None:
        map_keys = _default_mapping(type=type)

    for v in map_keys:
        network[v[0]] = network[v[1]].copy() * v[2]


def hydraulic_conductance(conn, pore_radii, throat_radii, conduit_length,
                          throat_density, throat_viscosity, rate,
                          gamma: float, C_0: float, E_0: float,
                          F: float, m: float, n: float) -> np.ndarray:
    r"""
    Computes the hydraulic conductance according to Eghbalmanesh and Fathiganjehlou (10.1016/j.ces.2023.119396)

    Parameters
    ----------
    conn: array_like
        connectivity of each throat, stored in an [Nt, 2] array, where the flow is positive
        if it flows from [:, 0] to [:, 1]
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

    Notes
    -----
    The conductance is provided as total value, including the conduit length, do not use with
    a gradient!
    """

    # preparing the arrays
    r_ij = throat_radii.copy().reshape((-1, 1)) * F
    mu = throat_viscosity.reshape((-1, 1))
    rho = throat_density.reshape((-1, 1))
    l_c = conduit_length.reshape((-1, 1))
    _rate = np.zeros(r_ij.shape, dtype=float) if rate is None else rate.copy()
    _rate.reshape(r_ij.shape)

    # flow has to be from pore i to pore j, so we have to reorder some
    # values in the case of reverse flow
    # Note, that the rate has to be provided in accordance with the connectivity
    # array!
    flow_rev = (_rate < 0.)

    _rate = np.abs(_rate)
    r_i = pore_radii[conn[:, 0]].reshape((-1, 1))
    r_j = pore_radii[conn[:, 1]].reshape((-1, 1))
    r_i[flow_rev], r_j[flow_rev] = r_j[flow_rev], r_i[flow_rev]

    # adapt throat radii if they exceed one of the pore radii
    mask = r_ij > r_i
    r_ij[mask] = r_i[mask]
    mask = r_ij > r_j
    r_ij[mask] = r_j[mask]

    # compute coefficients
    r_4 = r_ij**4
    Re_ij = 2 * rho * _rate/(math.pi * mu * r_ij)
    A_ij = 8 * mu  * l_c / (math.pi * r_4)
    C_ij = rho/(2 * math.pi**2 * r_4) * _rate * ((C_0/Re_ij)**n + 1./(2**n) * (1 - (r_ij/r_i)**2)**n)
    E_ij = rho/(2 * math.pi**2 * r_4) * _rate * ((E_0/Re_ij)**m + (1 - (r_ij/r_j)**2)**(2*m))
    G_ij = gamma * rho * _rate / (2 * math.pi**2) * (1/r_i**4 - 1/r_j**4)

    # correct for no flow
    noflow = _rate == 0.
    C_ij[noflow] = 0.
    E_ij[noflow] = 0.
    G_ij[noflow] = 0.

    # return conductance
    return (1./(A_ij + C_ij + E_ij - G_ij)).reshape((-1, 1))


class CalibratedConductance:
    def __init__(self,
                 network=None,
                 C_0:float = 26., E_0: float = 27., gamma = 1.,
                 F_hydro: float = 1., m: float = 1., n: float = 1.,
                 throat_conn = 'throat.conns',
                 throat_radii = 'throat.calibration_radius',
                 pore_radii = 'pore.calibration_radius',
                 throat_length = 'throat.calibration_length'):
        self.network = network
        self.C_0 = C_0
        self.E_0 = E_0
        self.F_hydro = F_hydro
        self.m = m
        self.n = n
        self.gamma = gamma
        self.l_tconn = throat_conn
        self.l_tradii = throat_radii
        self.l_pradii = pore_radii
        self.l_tlength = throat_length

    def PrepareNetwork(self, network=None, type: str = None, map_keys: list = None):
        r"""
        Conveniently prepares the provided or internally stored network so it can be used
        with the calibration functions

        Parameters
        ----------
        network: OpenPNM.network | dict
            an OpenPNM network or dictionary holding the keys which need to be mapped
        type: str
            optional parameter for specifying the type of extraction method used to generate the
            network, by default it will try to determine automatically if maximum ball or snow were
            used
        map_keys: list
            optional parameter to specify custom mappings for the calibration, where each entry in the
            list has to be of the form: [ label_target: str, label_source: str, factor_multiplicator: float]

        Returns
        -------
        Returns the parameters of the PrepareCalibratedValues function defined above, currently None
        """
        if network is not None:
            self.network = network
        if self.network is None:
            raise ValueError('Cannot prepare network, None was provided!')
        return PrepareCalibratedValues(network=network, type=type, map_keys=map_keys)

    def Hydraulic(self, viscosity, density, rate = None, throats: str|list[str]|np.ndarray = 'all', **kwargs) -> np.ndarray:
        r"""
        Computes the hydraulic conductance according to Eghbalmanesh and Fathiganjehlou (10.1016/j.ces.2023.119396)

        Parameters
        ----------
        viscosity: np.ndarray | float | int
            viscosity of each throat in Pa s
        density: np.ndarray | float | int
            fluid density of each throat in kg/m^3
        rate: np.ndarray
            rate of flow at each pore from conns[:, 0] to conns[:, 1], if none is provided it will
            be considered 0
        throats: str|list[str]|np.ndarray
            identifier, for which throats the conductance should be computed, compatible with the
            OpenPNM labels
        kwargs
            optional arguments, following possibilities currently apply
                mode:str -> mode for dealing with the provided labels, compatible with OpenPNM features
                            https://openpnm.org/modules/generated/openpnm.network.Network.throats.html

        Returns
        -------
        array of size [Nt,] with conductances in m^3/(Pa s)

        Notes
        -----
        The conductance is provided as total value, including the conduit length, do not use with
        a gradient!
        """
        if self.network is None:
            raise ValueError('The object has not been initialized for convenient access')

        throats_l = None
        if isinstance(throats, str) or isinstance(throats, list):
            mode = kwargs['mode'] if 'mode' in kwargs else 'and'
            throats_l = self.network.throats(labels=throats, mode=mode)
        else:
            throats_l = throats

        if throats_l.dtype != int:
            raise TypeError(f'The throats have to be provided as integer values! Found following type: {type(throats[0])}')

        Nt = self.network.num_throats()
        Nt_l = throats_l.size

        mu = viscosity if isinstance(viscosity, np.ndarray) else np.full((Nt, 1), fill_value=viscosity, dtype=float)
        rho = density if isinstance(density, np.ndarray) else np.full((Nt, 1), fill_value=density, dtype=float)
        if mu.size != Nt:
            raise ValueError(f'viscosity array is incompatible with number of throats! Provided array has shape: {mu.shape}')
        if rho.size != Nt:
            raise ValueError(f'density array is incompatible with number of throats! Provided array has shape: {rho.shape}')
        if rate is not None and rate.size != Nt:
            raise ValueError(f'rate array is incompatible with number of throats! Provided array has shape: {rate.shape}')

        g = hydraulic_conductance(conn=self.network[self.l_tconn],
                                  pore_radii=self.network[self.l_pradii],
                                  throat_radii=self.network[self.l_tradii],
                                  conduit_length=self.network[self.l_tlength],
                                  throat_viscosity=mu,
                                  throat_density=rho,
                                  rate=rate,
                                  gamma=self.gamma,
                                  C_0=self.C_0,
                                  E_0=self.E_0,
                                  F=self.F_hydro,
                                  m=self.m,
                                  n=self.n)

        if Nt_l < Nt:
            g = g[throats_l]

        return g
