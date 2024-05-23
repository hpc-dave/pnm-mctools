import openpnm as op
import scipy.linalg
import scipy.sparse
import spheres_and_cylinders as geo_model
import numpy as np
import scipy
from ToolSet import MulticomponentTools

def run(output:bool = True):

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
    c[1, 0] = 1.
    c[-2, 1] = 1.
    bc = {}

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

    # construct multiple upwind matrices (directed networks)
    c_up_float = mt._construct_upwind(fluxes=v[0])
    c_up_float_list = mt._construct_upwind(fluxes=v)
    c_up_array_single = mt._construct_upwind(fluxes=fluxes[:, 0])
    c_up_arrays_mult = mt.Upwind(fluxes=fluxes)

    # check the implementations
    if scipy.sparse.find(c_up_float_list - c_up_arrays_mult)[0].size > 0:
        raise ('matrices are inconsistent, check implementation')
    if scipy.sparse.find(c_up_float - c_up_array_single)[0].size > 0:
        # note that we don't do validation here
        raise ('matrices are inconsistent, check implementation')

    c_up = c_up_arrays_mult
    div = mt.Divergence(weights=A_flux)
    ddt = mt.DDT(dt=dt)

    J = ddt + div(fluxes, c_up)

    mass_init = np.sum(x)
    peakpos_init = (np.argmax(c[:, 0]), np.argmax(c[:, 1]))
    success = True
    for t in tsteps:
        x_old = x.copy()
        pos += 1

        G = J * x - ddt * x_old
        for i in range(max_iter):
            last_iter = i
            dx[:] = scipy.sparse.linalg.spsolve(J, -G).reshape(dx.shape)
            x = x + dx
            G = J * x - ddt * x_old
            G_norm = np.linalg.norm(np.abs(G), ord=2)
            if G_norm < tol:
                break
        if last_iter == max_iter - 1:
            print(f'WARNING: the maximum iterations ({max_iter}) were reached!')

        c = x.reshape(-1, Nc)
        mass = np.sum(x)
        peakpos = (np.argmax(c[:, 0]), np.argmax(c[:, 1]))
        mass_err = (mass_init - mass)/mass_init
        peakpos_err = [int(peakpos[n]-(peakpos_init[n] + pos * dt * v[n]/spacing)) for n in range(Nc)]
        success &= np.abs(mass_err) < 1e-12
        success &= np.max(np.abs(peakpos_err)) < 2
        if output:
            print(f'{t}/{len(tsteps)} - {time}: {last_iter + 1} it -\
                G [{G_norm:1.2e}] mass [{mass_err:1.2e}] peak-err [{peakpos_err}]')
        time += dt

    return success
