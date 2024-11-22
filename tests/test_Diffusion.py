import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

import openpnm as op                                      # noqa: E402
import scipy.linalg                                       # noqa: E402
import scipy.sparse                                       # noqa: E402
import models.const_spheres_and_cylinders as geo_model    # noqa: E402
import numpy as np                                        # noqa: E402
import scipy                                              # noqa: E402
import math                                               # noqa: E402
from ToolSet import MulticomponentTools                   # noqa: E402
import Operators as ops                                   # noqa: E402
import BoundaryConditions as bc                           # noqa: E402


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
    bc.set(mt=mt, id=0, label='left', bc={'prescribed': 1.})
    bc.set(mt=mt, id=1, label='right', bc={'prescribed': 1.})

    x = np.ndarray.flatten(c).reshape((c.size, 1))
    dx = np.zeros_like(x)

    dt = 0.01
    tsteps = range(1, int(1./dt))
    pos = 0
    tol = 1e-6
    max_iter = 10
    time = dt

    # need to provide div with some weights, namely an area
    # that the flux act on
    A_flux = np.zeros((network.Nt, 1), dtype=float) + network['pore.volume'][0]/spacing

    grad = ops.gradient(mt)
    div = ops.sum(mt)
    ddt = ops.ddt(mt, dt=dt)

    D = np.ones((network.Nt, Nc), dtype=float)
    J = ddt - div(A_flux, D, grad)

    def AnalyticalSolution(t, zeta):
        s_inf = 1
        s_tr = np.zeros_like(c[:, 0]).ravel()
        for n in range(100):
            npi = (n + 0.5) * math.pi
            s_tr += 2 * np.cos(n * math.pi)/npi * np.exp(-(npi)**2 * t) * np.cos(npi * zeta)
        return (s_inf - s_tr)

    zeta = np.asarray([network['pore.coords'][i][0] for i in range(network.Np)])
    zeta = zeta - zeta[0]
    zeta = zeta / (zeta[-1]+0.5*spacing)
    ana_sol = np.zeros_like(c)

    J = bc.apply(network=mt, A=J, type='Jacobian')
    success = True
    for t in tsteps:
        x_old = x.copy()
        pos += 1

        G = J * x - ddt * x_old
        G = bc.apply(network=mt, x=x, b=G, type='Defect')
        for i in range(max_iter):
            last_iter = i
            dx[:] = scipy.sparse.linalg.spsolve(J, -G).reshape(dx.shape)
            x = x + dx
            G = J * x - ddt * x_old
            G = bc.apply(network=mt, x=x, b=G, type='Defect')
            G_norm = np.linalg.norm(np.abs(G), ord=2)
            if G_norm < tol:
                break
        if last_iter == max_iter - 1:
            print(f'WARNING: the maximum iterations ({max_iter}) were reached!')

        c = x.reshape(c.shape)
        ana_sol[:, 0] = AnalyticalSolution(time, 1-zeta)
        ana_sol[:, 1] = np.flip(ana_sol[:, 0])
        err = ana_sol - c
        if pos > 10:
            success &= np.max(np.abs(err[:, 0])) < 1e-2
            success &= np.max(np.abs(err[:, 1])) < 1e-2
        if output:
            print(f'{t}/{len(tsteps)} - {time:1.2f}: {last_iter + 1} it [{G_norm:1.2e}]\
                err [{np.max(np.abs(err[:, 0])):1.2e} {np.max(np.abs(err[:, 1])):1.2e}]')
        time += dt

    return success


if __name__ == '__main__':
    run()
