import openpnm as op                                       # noqa: E402
import scipy, scipy.linalg, scipy.sparse                   # noqa: E401, E402
import pnm_mctools.models.const_spheres_and_cylinders as geo_model    # noqa: E402
import numpy as np                                         # noqa: E402
from pnm_mctools.ToolSet import MulticomponentTools        # noqa: E402
import pnm_mctools.Operators as ops                        # noqa: E402
import pnm_mctools.Interpolation as ip                     # noqa: E402
import pnm_mctools.BoundaryConditions as bc                # noqa: E402


def test_Outflow_convection(output: bool = True):
    Nx = 100
    Ny = 1
    Nz = 1
    Nc = 2
    spacing = 1./Nx

    # get network
    network = op.network.Cubic([Nx, Ny, Nz], spacing=spacing)

    # add geometry

    geo = geo_model.spheres_and_cylinders
    network.add_model_collection(geo, domain='all')
    network.regenerate_models()

    c = np.zeros((network.Np, Nc))

    mt = MulticomponentTools(network=network, num_components=Nc)
    bc.set(mt, id=0, label='left', bc={'value': 1})
    bc.set(mt, id=0, label='right', bc={'outflow'})
    bc.set(mt, id=1, label='left', bc={'outflow'})
    bc.set(mt, id=1, label='right', bc={'value': 1})

    x = np.ndarray.flatten(c).reshape((c.size, 1))
    dx = np.zeros_like(x)

    v = [0.1, -0.1]
    dt = 0.01
    tsteps = range(1, int(10./dt))
    sol = np.zeros_like(c)
    sol = np.tile(sol, reps=len(tsteps)+1)
    pos = 0
    tol = 1e-6
    max_iter = 10
    time = dt

    # need to provide div with some weights, namely an area
    # that the flux act on
    A_flux = np.full((network.Nt, 1), fill_value=network['pore.volume'][0]/spacing, dtype=float)

    fluxes = np.zeros((network.Nt, 2), dtype=float)
    fluxes[:, 0] = v[0]
    fluxes[:, 1] = v[1]

    c_up = ip.upwind(mt, fluxes=fluxes)
    sum = ops.sum(mt)
    ddt = ops.ddt(mt, dt=dt)

    J = ddt + sum(A_flux, fluxes, c_up)
    J = bc.apply(mt, A=J, x=x)

    mass_init = np.sum(c[1:-1, :] * network['pore.volume'][1:-1].reshape(network.Np-2, 1), axis=0)

    A_flux_mult = np.append(A_flux, A_flux)
    flux_at_faces = c_up.multiply(fluxes.reshape(-1, 1)).multiply(A_flux_mult.reshape(-1, 1))

    inflow = np.zeros((2))
    outflow = np.zeros((2))
    for t in tsteps:
        x_old = x.copy()
        pos += 1

        G = J * x - ddt * x_old
        G = bc.apply(mt, b=G, x=x, type='Defect')
        for i in range(max_iter):
            last_iter = i
            dx[:] = scipy.sparse.linalg.spsolve(J, -G).reshape(dx.shape)
            x = x + dx
            G = J * x - ddt * x_old
            G = bc.apply(mt, b=G, x=x, type='Defect')
            G_norm = np.linalg.norm(np.abs(G), ord=2)
            if G_norm < tol:
                break
        if last_iter == max_iter - 1:
            print(f'WARNING: the maximum iterations ({max_iter}) were reached!')

        c = x.reshape(-1, Nc)
        mass_tot = np.sum(c[1:-1, :] * network['pore.volume'][1:-1].reshape(network.Np-2, 1), axis=0)
        fluxes_num = (flux_at_faces * c.reshape(-1, 1)).reshape(-1, 2)
        flux_in, flux_out = fluxes_num[0, :], fluxes_num[-1, :]
        inflow += flux_in * dt
        outflow += flux_out * dt
        mass_err = (inflow-outflow) / (mass_tot - mass_init) - 1
        assert np.max(np.abs(mass_err)) < 1e-13
        if output:
            print(f'{t}/{len(tsteps)} - {time:1.2}: {last_iter + 1} it -\
                G [{G_norm:1.2e}] mass [{mass_err[0]:1.2e} {mass_err[1]:1.2e}]')
        time += dt
