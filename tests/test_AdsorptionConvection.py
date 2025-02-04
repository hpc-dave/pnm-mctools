import openpnm as op                                                    # noqa: E402
import scipy.linalg                                                     # noqa: E402
import scipy.sparse                                                     # noqa: E402
import numpy as np                                                      # noqa: E402
import scipy                                                            # noqa: E402
from pnm_mctools.ToolSet import MulticomponentTools                     # noqa: E402
import pnm_mctools.models.const_spheres_and_cylinders as geo_model      # noqa: E402
import pnm_mctools.Operators as ops                                     # noqa: E402
import pnm_mctools.Adsorption as ads                                    # noqa: E402
import pnm_mctools.Interpolation as ip                                  # noqa: E402
import pnm_mctools.BoundaryConditions as bc                             # noqa: E402


def test_convection_linear(output: bool = True):
    Nx = 10
    Ny = 1
    Nz = 1
    Nc = 2
    spacing = 1./Nx
    K_ads = 0.1
    c_0 = 0.
    c_pulse = 1.
    dt = 1.

    # get network
    network = op.network.Cubic([Nx, Ny, Nz], spacing=spacing)
    bc_out = 'right'

    # add geometry
    geo = geo_model.spheres_and_cylinders
    network.add_model_collection(geo, domain='all')
    network.regenerate_models()

    # adsorption data
    id_ads = 0

    def K_lin(c_f):
        return np.full((c_f.shape[0], 1), fill_value=K_ads)

    a_V = np.full((network.Np, 1), fill_value=5., dtype=float)

    c = np.zeros((network.Np, Nc))
    c[:, id_ads] = c_0
    c[0, id_ads] = c_pulse

    c_old = c.copy()

    mt = MulticomponentTools(network=network, num_components=Nc)

    bc.set(mt, id=id_ads, label=bc_out, bc={'outflow'})

    x = c.reshape((-1, 1)).copy()
    dx = np.zeros_like(x)

    v = 0.1
    dt = spacing/v
    t_end = 5.
    tsteps = range(1, int(t_end/dt))

    tol = 1e-12
    max_iter = 100

    J_ads, G_ads = ads.single_linear(c=c, c_old=c_old,
                                     K_func=K_lin,
                                     dt=dt,
                                     component_id=id_ads,
                                     network=mt,
                                     a_v=a_V)

    K_init = K_lin(c[:, id_ads])
    m_0 = c.copy()
    m_0[:, id_ads] *= (1. + K_init*a_V).reshape((-1))
    m_0 *= network['pore.volume'].reshape((-1, 1))

    x_old = x.copy()

    A_conv = np.zeros((network.Nt, 1), dtype=float) + network['pore.volume'][0]/spacing
    c_up = ip.upwind(network=mt, rates=v)
    ddt = ops.ddt(mt, dt=dt)
    sum = ops.sum(network=mt)
    J_conv = sum(A_conv, v, c_up)

    pulse_out = np.zeros_like(tsteps, dtype=float)
    for n in range(len(tsteps)):
        x_old = x.copy()
        J_ads, G_ads = ads.single_linear(c=x.reshape(c.shape), c_old=x_old.reshape(c.shape),
                                         K_func=K_lin,
                                         dt=dt,
                                         component_id=id_ads,
                                         network=mt,
                                         a_v=a_V)
        J = ddt + J_conv + J_ads
        G = ddt * x - ddt * x_old + J_conv * x + G_ads

        for i in range(max_iter):
            last_iter = i
            dx = scipy.sparse.linalg.spsolve(J, -G).reshape(dx.shape)
            x = x + dx
            c = x.reshape(c.shape).copy()

            J_ads, G_ads = ads.single_linear(c=x.reshape(c.shape), c_old=x_old.reshape(c.shape),
                                             K_func=K_lin,
                                             dt=dt,
                                             component_id=id_ads,
                                             network=mt,
                                             a_v=a_V,
                                             stype='Jacobian')

            G = ddt * x - ddt * x_old + J_conv * x + G_ads
            G_norm = np.linalg.norm(np.abs(G), ord=2)
            if G_norm < tol:
                break
        assert last_iter < (max_iter-1)
        pulse_out[n] = c[-2, id_ads]


def test_convection_Langmuir(output: bool = True):
    Nx = 50
    Ny = 1
    Nz = 1
    Nc = 2
    spacing = 1./Nx
    K_ads = 1.
    y_max = 1.
    c_0 = 0.
    c_pulse = 1.
    dt = 1.

    # get network
    network = op.network.Cubic([Nx, Ny, Nz], spacing=spacing)
    bc_out = 'right'

    # add geometry
    geo = geo_model.spheres_and_cylinders
    network.add_model_collection(geo, domain='all')
    network.regenerate_models()

    # adsorption data
    id_ads = 0

    a_V = np.full((network.Np, 1), fill_value=1., dtype=float)

    mt = MulticomponentTools(network=network, num_components=Nc)

    bc.set(mt, id=id_ads, label=bc_out, bc={'outflow'})

    v = 0.1
    dt = spacing / v
    t_end = 50.
    tsteps = range(1, int(t_end/dt))

    tol = 1e-12
    max_iter = 100

    pulse_dict = {}

    def theta_Langmuir(c_f):
        return ads.Langmuir(c_f, K_ads, y_max=y_max)

    def Adsorption(c_f, c_f_old):
        J_ads, G_ads = ads.multi_component(c=c_f, c_old=c_f_old,
                                           theta_func=theta_Langmuir,
                                           dt=dt,
                                           component_id=id_ads,
                                           network=mt,
                                           a_v=a_V,
                                           Vp='pore.volume')
        return J_ads, G_ads

    c = np.zeros((network.Np, Nc))
    c[:, id_ads] = c_0
    c[0, id_ads] = c_pulse*Nx/10

    x = c.reshape((-1, 1)).copy()
    dx = np.zeros_like(x)

    x_old = x.copy()

    A_conv = np.zeros((network.Nt, 1), dtype=float) + network['pore.volume'][0]/spacing
    c_up = ip.upwind(network=mt, rates=v)
    ddt = ops.ddt(mt, dt=dt)
    sum = ops.sum(network=mt)
    J_conv = sum(A_conv, v, c_up)

    pulse_out = np.zeros_like(tsteps, dtype=float)
    for n in range(len(tsteps)):
        x_old = x.copy()
        J_ads, G_ads = Adsorption(c_f=x.reshape(c.shape), c_f_old=x_old.reshape(c.shape))
        J = ddt + J_conv + J_ads
        G = ddt * x - ddt * x_old + J_conv * x + G_ads

        for i in range(max_iter):
            last_iter = i
            dx = scipy.sparse.linalg.spsolve(J, -G).reshape(dx.shape)
            x = x + dx
            c = x.reshape(c.shape).copy()

            J_ads, G_ads = Adsorption(c_f=x.reshape(c.shape), c_f_old=x_old.reshape(c.shape))
            G = ddt * x - ddt * x_old + J_conv * x + G_ads
            G_norm = np.linalg.norm(np.abs(G), ord=2)
            if G_norm < tol:
                break
        assert last_iter < (max_iter-1)
        pulse_out[n] = c[-2, id_ads]

    pulse_dict[f'K={K_ads} - ymax={y_max}'] = pulse_out.copy()
