import openpnm as op                                                  # noqa: E402
import scipy, scipy.linalg, scipy.sparse                              # noqa: E401, E402
import pnm_mctools.models.const_spheres_and_cylinders as geo_model    # noqa: E402
import numpy as np                                                    # noqa: E402
from pnm_mctools.ToolSet import MulticomponentTools                   # noqa: E402
import pnm_mctools.Operators as ops                                   # noqa: E402
import pnm_mctools.Interpolation as ip                                # noqa: E402
import pnm_mctools.BoundaryConditions as bc                           # noqa: E402
from matplotlib import pyplot as plt                                   # noqa: E402
try:
    import pandas as pd                                                  # noqa: E402
except ImportError:
    pd = None


def analytical_solution(tau, zeta, Pe):
    f1 = -scipy.special.erfc((Pe * tau - zeta)/(2. * np.sqrt(tau)))
    f2 = np.exp(Pe * zeta) * scipy.special.erfc((Pe * tau + zeta)/(2. * np.sqrt(tau)))
    return 0.5 * (f1 + f2 + 2)


def test_DiffusionConvection(output: bool = True, file_output: bool = False):
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
    v = 0.001
    D_bin = 1e-3

    mt = MulticomponentTools(network=network)
    bc.set(mt, label='left', bc=1.)
    bc.set(mt, label='right', bc={'outflow'})

    x = np.ndarray.flatten(c).reshape((c.size, 1))
    dx = np.zeros_like(x)

    dt = 0.1
    tsteps = range(1, int(10./dt))
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

    D = np.zeros((network.Nt, 1), dtype=float) + D_bin
    J = ddt - sum(A_flux, D, grad) + sum(A_flux, v, c_up)

    J = bc.apply(mt, A=J)

    Pe = v/D_bin

    err = 0

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

        zeta = network['pore.coords'][:, 0] - network['pore.coords'][0, 0]
        sol_ana = analytical_solution(tau=time*D_bin, zeta=zeta, Pe=Pe)

        err = np.sum(np.abs(x - sol_ana))/Nx

        if output:
            print(f'{t}/{len(tsteps)} - {time}: {last_iter + 1} it [{G_norm}] error: {err}')
        if file_output:
            if pd is None:
                raise ImportError('pandas is not installed, cannot save output to csv')
            df = op.io.network_to_pandas(network)
            df = pd.concat([*df.values(), pd.DataFrame({'c_num': x.reshape(-1), 'c_ana': sol_ana.reshape(-1)})], axis=1)
            df.to_csv(f'output_DiffusionConvection_{t}.csv')
        time += dt

    assert err < 20., f'Error is too high: {err}, maximum allowed is 20'
    print('DiffusionConvection test does not have a success criteria yet!')
