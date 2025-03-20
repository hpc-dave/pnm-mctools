from pnm_mctools import Calibration as CB
from math import pi
import numpy as np
import openpnm as op


def test_calibration_fathiganjehlou():
    # define basic parameters
    Np = 10
    Nt = 9

    # radii
    pore_radii = np.arange(1, Np+1, dtype=float)
    throat_radii = np.arange(0.01, (Nt+1)*0.01, 0.01, dtype=float)
    # connectivity array
    conn = np.hstack([np.arange(Nt).reshape((-1, 1)), np.arange(1, Np).reshape((-1, 1))])
    # other properties
    conduit_length = np.full_like(throat_radii, fill_value=0.11)
    throat_density = np.full_like(throat_radii, fill_value=0.22)
    throat_viscosity = np.full_like(throat_radii, fill_value=0.33)
    rate = np.full_like(conduit_length, fill_value=1.)

    # calibration values
    C_0 = 27.
    E_0 = 26.
    gamma = 1.
    F = 1.
    m = 0.
    n = 0.

    g_h = CB.hydraulic_conductance_fathiganjehlou(conn=conn,
                                                  pore_radii=pore_radii, throat_radii=throat_radii,
                                                  conduit_length=conduit_length,
                                                  throat_density=np.zeros_like(throat_density),
                                                  throat_viscosity=throat_viscosity,
                                                  rate=rate,
                                                  gamma=gamma, C_0=C_0, E_0=E_0, F=F, m=m, n=n)

    A = ((pi * (throat_radii/F)**4)/(8. * throat_viscosity * rate * conduit_length)).reshape((-1, 1))
    assert np.all((A-g_h) < 1e-16)

    r = 0.1
    g_h = CB.hydraulic_conductance_fathiganjehlou(conn=conn,
                                                  pore_radii=np.full_like(pore_radii, fill_value=r),
                                                  throat_radii=np.full_like(throat_radii, fill_value=r),
                                                  conduit_length=np.zeros_like(conduit_length),
                                                  throat_density=throat_density,
                                                  throat_viscosity=throat_viscosity,
                                                  rate=rate,
                                                  gamma=gamma, C_0=0., E_0=E_0, F=F, m=1.5, n=1.)

    Re_ij = (2*throat_density*np.abs(rate)/(pi*throat_viscosity*r))
    E = ((2 * pi**2 * r**4)/(throat_density*np.abs(rate)) * (Re_ij/E_0)**1.5).reshape((-1, 1))

    assert np.all((E-g_h) < 1e-16)

    g_h = CB.hydraulic_conductance_fathiganjehlou(conn=conn,
                                                  pore_radii=np.full_like(pore_radii, fill_value=r),
                                                  throat_radii=np.full_like(throat_radii, fill_value=r),
                                                  conduit_length=np.zeros_like(conduit_length),
                                                  throat_density=throat_density,
                                                  throat_viscosity=throat_viscosity,
                                                  rate=rate,
                                                  gamma=gamma, C_0=C_0, E_0=0., F=F, m=1., n=1.4)

    Re_ij = (2*throat_density*np.abs(rate)/(pi*throat_viscosity*r))
    C = ((2 * pi**2 * r**4)/(throat_density*np.abs(rate)) * (Re_ij/C_0)**1.4).reshape((-1, 1))

    assert np.all((C-g_h) < 1e-16)

    g_h = CB.hydraulic_conductance_fathiganjehlou(conn=conn,
                                                  pore_radii=pore_radii,
                                                  throat_radii=np.full_like(throat_radii, fill_value=r),
                                                  conduit_length=np.zeros_like(conduit_length),
                                                  throat_density=throat_density,
                                                  throat_viscosity=throat_viscosity,
                                                  rate=rate,
                                                  gamma=gamma, C_0=C_0, E_0=0., F=1., m=0., n=0.)

    Re_ij = (2*throat_density*np.abs(rate)/(pi*throat_viscosity*r))
    G_ij = gamma*(1./pore_radii[:-1]**4 - 1./pore_radii[1:]**4)
    G = ((2 * pi**2)/(throat_density*np.abs(rate)) / (4/r**4 + - G_ij)).reshape((-1, 1))

    assert np.all((G-g_h) < 1e-16)


def test_calibration_fathiganjehlou_wrapper():
    # define basic parameters
    Np = 10
    Nt = 9

    # radii
    pore_radii = np.arange(1, Np+1, dtype=float)
    throat_radii = np.arange(0.01, (Nt+1)*0.01, 0.01, dtype=float)
    # connectivity array
    conn = np.hstack([np.arange(Nt).reshape((-1, 1)), np.arange(1, Np).reshape((-1, 1))])
    # other properties
    conduit_length = np.full_like(throat_radii, fill_value=0.11)
    throat_density = np.full_like(throat_radii, fill_value=0.22)
    throat_viscosity = np.full_like(throat_radii, fill_value=0.33)
    rate = np.full_like(conduit_length, fill_value=1.)

    coord = np.cumsum(np.hstack([[0.], conduit_length])).reshape((-1, 1))
    coord = np.hstack([coord, np.zeros_like(coord), np.zeros_like(coord)])

    # calibration values
    C_0 = 27.
    E_0 = 26.
    gamma = 1.
    F = 1.
    m = 0.
    n = 0.

    network = op.network.Network(conns=conn, coords=coord)
    network['throat.calibration_radius'] = throat_radii
    network['throat.calibration_length'] = conduit_length
    network['pore.calibration_radius'] = pore_radii
    cond = CB.ConductanceFathiganjehlou(network=network, C_0=C_0, E_0=E_0, gamma=gamma, F_hydro=F, m=m, n=n)

    g_h = cond.Hydraulic(viscosity=throat_viscosity, density=np.zeros_like(throat_density), rate=rate)

    A = ((pi * (throat_radii/F)**4)/(8. * throat_viscosity * rate * conduit_length)).reshape((-1, 1))
    assert np.all((A-g_h) < 1e-16)


if __name__ == "__main__":
    test_calibration_fathiganjehlou_wrapper()
    print('finished')
