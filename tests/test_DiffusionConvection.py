import openpnm as op                                                  # noqa: E402
import scipy, scipy.linalg, scipy.sparse                              # noqa: E401, E402
import pnm_mctools.models.const_spheres_and_cylinders as geo_model    # noqa: E402
import numpy as np                                                    # noqa: E402
from pnm_mctools.ToolSet import MulticomponentTools                   # noqa: E402
import pnm_mctools.Operators as ops                                   # noqa: E402
import pnm_mctools.Interpolation as ip                                # noqa: E402
import pnm_mctools.BoundaryConditions as bc                           # noqa: E402


def test_DiffusionConvection(output: bool = True):
    Nx = 100
    Ny = 1
    Nz = 1
    spacing = 1./Nx

    # get network
    network = op.network.Cubic([Nx, Ny, Nz], spacing=spacing)

    # add geometry
    geo = geo_model.spheres_and_cylinders
    network.add_model_collection(geo, domain='all')
    network.regenerate_models()

    c = np.zeros((network.Np, 1))
    v = 0.1

    mt = MulticomponentTools(network=network)
    bc.set(mt, label='left', bc=1.)
    bc.set(mt, label='right', bc={'outflow'})

    x = np.ndarray.flatten(c).reshape((c.size, 1))
    dx = np.zeros_like(x)

    dt = 0.01
    tsteps = range(1, int(1./dt))
    pos = 0
    tol = 1e-6
    max_iter = 10
    time = dt

    # need to provide sum with some weights, namely an area
    # that the flux acts on
    A_flux = np.zeros((network.Nt, 1), dtype=float) + network['pore.volume'][0]/spacing

    grad = ops.gradient(mt)
    c_up = ip.upwind(mt, fluxes=v)
    sum = ops.sum(mt)
    ddt = ops.ddt(mt, dt=dt)

    D = np.zeros((network.Nt, 1), dtype=float) + 1e-3
    J = ddt - sum(A_flux, D, grad) + sum(A_flux, v, c_up)

    J = bc.apply(mt, A=J)

    for t in tsteps:
        x_old = x.copy()
        pos += 1

        G = J * x - ddt * x_old
        G = bc.apply(mt, x=x, b=G, type='Defect')
        for i in range(max_iter):
            last_iter = i
            dx[:] = scipy.sparse.linalg.spsolve(J, -G).reshape(dx.shape)
            x = x + dx
            G = J * x - ddt * x_old
            G = bc.apply(mt, x=x, b=G, type='Defect')
            G_norm = np.linalg.norm(np.abs(G), ord=2)
            if G_norm < tol:
                break
        if last_iter == max_iter - 1:
            print(f'WARNING: the maximum iterations ({max_iter}) were reached!')

        if output:
            print(f'{t}/{len(tsteps)} - {time}: {last_iter + 1} it [{G_norm}]')
        time += dt

    print('DiffusionConvection test does not have a success criteria yet!')
