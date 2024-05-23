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
    Nc = 2
    spacing = 1./Nx
    rate_in = 0.1   # unit is [arbitrary]/s

    # get network
    network = op.network.Cubic([Nx, Ny, Nz], spacing=spacing)

    # add geometry

    geo = geo_model.spheres_and_cylinders
    network.add_model_collection(geo, domain='all')
    network.regenerate_models()

    c = np.zeros((network.Np, Nc))
    c_old = c.copy()
    bc_0, bc_1 = {}, {}
    bc_0['left'] = {'rate': rate_in}
    bc_1['right'] = {'rate': rate_in}
    bc = [bc_0, bc_1]

    mt = MulticomponentTools(network=network, num_components=Nc, bc=bc)

    x = np.ndarray.flatten(c).reshape((c.size, 1))
    dx = np.zeros_like(x)

    v = [0.1, -0.1]
    dt = 0.01
    tsteps = range(1, int(5./dt))
    sol = np.zeros_like(c)
    sol = np.tile(sol, reps=len(tsteps)+1)
    pos = 0
    tol = 1e-6
    max_iter = 10
    time = dt

    # need to provide div with some weights, namely an area
    # that the flux act on
    A_flux = np.zeros((network.Nt, 1), dtype=float) + network['pore.volume'][0]/spacing

    fluxes = np.zeros((network.Nt, 2), dtype=float)
    fluxes[:, 0] = v[0]
    fluxes[:, 1] = v[1]

    c_up = mt.Upwind(fluxes=fluxes)
    div = mt.Divergence(weights=A_flux)
    ddt = mt.DDT(dt=dt)

    J = ddt + div(fluxes, c_up)
    J = mt.ApplyBC(A=J, x=x)

    mass_init = np.sum(c * network['pore.volume'].reshape(network.Np, 1), axis=0)

    success = True
    for t in tsteps:
        x_old = x.copy()
        pos += 1

        G = J * x - ddt * x_old
        G = mt.ApplyBC(b=G, x=x, type='Defect')
        for i in range(max_iter):
            last_iter = i
            dx[:] = scipy.sparse.linalg.spsolve(J, -G).reshape(dx.shape)
            x = x + dx
            G = J * x - ddt * x_old
            G = mt.ApplyBC(b=G, x=x, type='Defect')
            G_norm = np.linalg.norm(np.abs(G), ord=2)
            if G_norm < tol:
                break
        if last_iter == max_iter - 1:
            print(f'WARNING: the maximum iterations ({max_iter}) were reached!')

        c = x.reshape(-1, Nc)
        mass_tot = np.sum(c * network['pore.volume'].reshape(network.Np, 1), axis=0)
        mass_in = rate_in * time
        mass_err = (mass_tot - mass_init)/mass_in - 1
        success = np.max(np.abs(mass_err)) < 1e-13
        if output:
            print(f'{t}/{len(tsteps)} - {time}: {last_iter + 1} it -\
                G [{G_norm:1.2e}] mass [{mass_err[0]:1.2e} {mass_err[1]:1.2e}]')
        time += dt

    return success
