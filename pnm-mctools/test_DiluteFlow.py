import openpnm as op
import scipy.linalg
import scipy.sparse
import spheres_and_cylinders as geo_model
import numpy as np
import scipy
from ToolSet import MulticomponentTools


def run(output: bool = True):
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

    # flow properties
    water = op.phase.Water(network=network)
    water.add_model(propname='throat.hydraulic_conductance',
                    model=op.models.physics.hydraulic_conductance.generic_hydraulic)

    sf = op.algorithms.StokesFlow(network=network, phase=water)
    sf.set_value_BC(pores=network.pores('left'), values=1.1e5)
    sf.set_value_BC(pores=network.pores('right'), values=1e5)
    sf.run()

    c = np.zeros((network.Np, 1))
    bc = {}
    bc['left'] = {'prescribed': 1.}
    bc['right'] = {'outflow': None}

    mt = MulticomponentTools(network=network, bc=bc)

    x = np.ndarray.flatten(c).reshape((c.size, 1))
    dx = np.zeros_like(x)

    dt = 0.01
    tsteps = range(1, int(1./dt))
    pos = 0
    tol = 1e-6
    max_iter = 10
    time = dt

    # need to provide div with some weights, namely an area
    # that the flux acts on
    A_flux = np.zeros((network.Nt, 1), dtype=float) + network['pore.volume'][0]/spacing
    fluid_flux = sf.rate(throats=network.throats('all'), mode='single')
    grad = mt.Gradient()
    c_up = mt.Upwind(fluxes=fluid_flux)
    div = mt.Divergence()
    ddt = mt.DDT(dt=dt)

    D = np.full((network.Nt, 1), fill_value=1e-6, dtype=float)

    J = ddt - div(D, A_flux, grad) + div(fluid_flux, c_up)

    J = mt.ApplyBC(A=J)
    success = True
    for t in tsteps:
        x_old = x.copy()
        pos += 1

        G = J * x - ddt * x_old
        G = mt.ApplyBC(x=x, b=G, type='Defect')
        for i in range(max_iter):
            last_iter = i
            dx[:] = scipy.sparse.linalg.spsolve(J, -G).reshape(dx.shape)
            x = x + dx
            G = J * x - ddt * x_old
            G = mt.ApplyBC(x=x, b=G, type='Defect')
            G_norm = np.linalg.norm(np.abs(G), ord=2)
            if G_norm < tol:
                break
        if last_iter == max_iter - 1:
            print(f'WARNING: the maximum iterations ({max_iter}) were reached!')

        if output:
            print(f'{t}/{len(tsteps)} - {time}: {last_iter + 1} it [{G_norm}]')
        time += dt

    print('DiluteFlow does not have a success criteria yet')
    return success
