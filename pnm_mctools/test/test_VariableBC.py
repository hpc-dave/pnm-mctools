import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

import openpnm as op                                       # noqa: E402
import scipy, scipy.linalg, scipy.sparse                   # noqa: E401, E402
import models.const_spheres_and_cylinders as geo_model     # noqa: E402
import numpy as np                                         # noqa: E402
from ToolSet import MulticomponentTools                    # noqa: E402
import Operators as ops                                    # noqa: E402
import Interpolation as ip                                 # noqa: E402
import BoundaryConditions as bc                            # noqa: E402


def run(output: bool = True):
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
    bc.set(mt, id=0, label='left', bc={'value': 1.})
    bc.set(mt, id=1, label='right', bc={'value': 1.})

    x = np.ndarray.flatten(c).reshape((c.size, 1))
    dx = np.zeros_like(x)

    v = [0.1, -0.1]
    dt = 0.01
    tsteps = range(1, int(5./dt))
    tol = 1e-6
    max_iter = 10
    time = dt

    # need to provide div with some weights, namely an area
    # that the flux act on
    A_flux = np.zeros((network.Nt, 1), dtype=float) + network['pore.volume'][0]/spacing

    fluxes = np.zeros((network.Nt, 2), dtype=float)
    fluxes[:, 0] = v[0]
    fluxes[:, 1] = v[1]

    c_up = ip.upwind(mt, fluxes=fluxes)
    sum = ops.sum(mt)
    ddt = ops.ddt(mt, dt=dt)

    J = ddt + sum(A_flux, fluxes, c_up)
    J = bc.apply(mt, A=J, x=x)

    success = True
    for n in range(len(tsteps)):
        x_old = x.copy()
        if n == 100:
            # update BC
            bc.set(mt, id=0, label='left', bc={'value': 0.})
            bc.set(mt, id=1, label='right', bc={'value': 0.})
            # no need to update the matrix

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
        success &= c[0, 0] == 1. if n < 100 else c[0, 0] == 0.
        success &= c[-1, 1] == 1. if n < 100 else c[-1, 1] == 0.
        if output:
            print(f'{n}/{len(tsteps)} - {time}: {last_iter + 1} it -\
                G [{G_norm:1.2e}]')
        time += dt

    return success


if __name__ == "__main__":
    success = run()

    if success:
        print('success')
    else:
        print('fail')
