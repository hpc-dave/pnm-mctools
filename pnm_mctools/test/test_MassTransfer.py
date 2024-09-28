import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

import openpnm as op                                       # noqa: E402
import scipy, scipy.linalg, scipy.sparse                   # noqa: E401, E402
import testing.const_spheres_and_cylinders as geo_model    # noqa: E402
import numpy as np                                         # noqa: E402
from ToolSet import MulticomponentTools                    # noqa: E402

Nx = 10
Ny = 100000
Nz = 1
dx = 1./Nx

# get network
network = op.network.Cubic([Nx, Ny, Nz], spacing=dx)

# add geometry

geo = geo_model.spheres_and_cylinders
network.add_model_collection(geo, domain='all')
network.regenerate_models()

c = np.zeros((network.Np, 1))
c_old = c.copy()
bc = {}
bc['left'] = {'prescribed': 1.}
bc['right'] = {'prescribed': 0.}

x = np.ndarray.flatten(c).reshape((c.size, 1))
dx = np.zeros_like(x)
x_old = x.copy()

mt = MulticomponentTools(network=network, num_components=1, bc=bc)
grad = mt.get_gradient_matrix()
sum = mt.get_sum()
ddt = mt.get_ddt(dt=0.0001)
D = np.ones((network.Nt, 1), dtype=float)

J = ddt - sum(D, grad)
J = mt.apply_bc(A=J)

for i in range(10):
    # timesteps
    x_old = x.copy()

    G = J * x - ddt * x_old
    G = mt.apply_bc(x=x, b=G, type='Defect')

    for n in range(10):
        # iterations (should not take more than one!)
        dx[:] = scipy.sparse.linalg.spsolve(J, -G).reshape(dx.shape)
        x = x + dx
        G = J * x - ddt * x_old
        G = mt.apply_bc(x=x, b=G, type='Defect')
        G_norm = np.linalg.norm(np.abs(G), ord=2)
        if G_norm < 1e-6:
            break

# define a phase
phase = op.phase.Air(network=network)

# add physics model
phys = op.models.collections.physics.basic
del phys['throat.entry_pressure']
phase.add_model_collection(phys)
phase.regenerate_models()

# define algorithm
# alg = op.algorithms.FickianDiffusion(network=network, phase=phase)
alg = op.algorithms.TransientFickianDiffusion(network=network, phase=phase)

# define BC
inlet = network.pores('left')
outlet = network.pores('right')
c_in, c_out = [1, 0]
alg.set_value_BC(pores=inlet, values=c_in)
alg.set_value_BC(pores=outlet, values=c_out)

x0 = np.zeros_like(network['pore.diameter'])
alg.run(x0=x0, tspan=[0, 1])

print('finished')
