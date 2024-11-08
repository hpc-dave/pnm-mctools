import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

import openpnm as op                                        # noqa: E402
import scipy.linalg                                         # noqa: E402
import scipy.sparse                                         # noqa: E402
import numpy as np                                          # noqa: E402
import scipy                                                # noqa: E402
from ToolSet import MulticomponentTools                     # noqa: E402
import Operators as ops                                     # noqa: E402
from Adsorption import AdsorptionSingleComponent            # noqa: E402
from Adsorption import Linear, Langmuir, Freundlich         # noqa: E402
import Adsorption as ads                                    # noqa: E402


def run(output: bool = True):
    success = True
    success &= run_single_linear(output)
    if not success:
        print('Linear failed')
    # success &= run_Linear(output)
    # if not success:
    #     print('Linear failed')
    # success &= run_Langmuir(output)
    # if not success:
    #     print('Linear failed')
    # success &= run_Freundlich(output)
    # if not success:
    #     print('Linear failed')
    return success


def run_single_linear(output: bool = True):
    Nx = 10
    Ny = 1
    Nz = 1
    Nc = 2
    spacing = 1./Nx
    K_ads = 1.
    c_0 = 1.
    source = 1.
    dt = 1.

    # get network
    network = op.network.Cubic([Nx, Ny, Nz], spacing=spacing)

    # add geometry
    network.add_model_collection(op.models.collections.geometry.spheres_and_cylinders, domain='all')
    network.regenerate_models()

    # adsorption data
    id_ads = 0

    def K_lin(c_f):
        return np.full((c_f.shape[0], 1), fill_value=K_ads)

    a_V = np.full((network.Np, 1), fill_value=5., dtype=float)

    c = np.zeros((network.Np, Nc))
    c[:, id_ads] = c_0

    c_old = c.copy()

    mt = MulticomponentTools(network=network, num_components=Nc)

    x = c.reshape((-1, 1)).copy()
    dx = np.zeros_like(x)

    tol = 1e-12
    max_iter = 100
    ddt = ops.ddt(mt, dt=dt)

    J_ads, G_ads = ads.single_linear(c=c, c_old=c_old,
                                     K_func=K_lin,
                                     dt=dt,
                                     component_id=id_ads,
                                     network=mt,
                                     a_v=a_V)
    G_source = np.zeros_like(c)
    G_source[:, id_ads] = -source * network['pore.volume'].reshape((-1))
    G_source = G_source.reshape((-1, 1))

    success = True

    K_init = K_lin(c[:, id_ads])
    m_0 = c.copy()
    m_0[:, id_ads] *= (1. + K_init*a_V).reshape((-1))
    m_0 *= network['pore.volume'].reshape((-1, 1))

    x_old = x.copy()
    J = ddt + J_ads
    G = ddt * x - ddt * x_old + G_ads + G_source

    for i in range(max_iter):
        last_iter = i

        dx = scipy.sparse.linalg.spsolve(J, -G).reshape(dx.shape)
        x = x + dx
        c = x.reshape(c.shape).copy()

        G_ads = ads.single_linear(c=c, c_old=c_old,
                                  K_func=K_lin,
                                  dt=dt,
                                  component_id=id_ads,
                                  network=mt,
                                  a_v=a_V,
                                  stype='Defect')

        G = ddt * x - ddt * x_old + G_ads + G_source
        G_norm = np.linalg.norm(np.abs(G), ord=2)
        if G_norm < tol:
            break
    if last_iter == max_iter - 1:
        print(f'WARNING: the maximum iterations ({max_iter}) were reached!')

    K_final = K_lin(c[:, id_ads]).reshape((-1, 1))
    m = c.copy()
    m[:, id_ads] *= (1. + K_final*a_V).reshape((-1))
    m *= network['pore.volume'].reshape((-1, 1))
    m_s = np.zeros_like(c)
    m_s[:, id_ads] = source * network['pore.volume'].reshape((-1))
    err = np.sum(m - (m_0 + m_s))/np.sum(m_0)
    success &= err < 1e-8

    if output:
        print(f'{last_iter + 1} it [{G_norm:1.2e}] mass-loss [{err:1.2e}]')
    return success


def run_Linear(output: bool = True):
    Nx = 10
    Ny = 1
    Nz = 1
    Nc = 3
    spacing = 1./Nx

    # get network
    network = op.network.Cubic([Nx, Ny, Nz], spacing=spacing)

    # add geometry
    network.add_model_collection(op.models.collections.geometry.spheres_and_cylinders, domain='all')
    network.regenerate_models()

    # adsorption data
    ads = 0
    dil = 1

    def y_f(c_f, c_ads):
        return Linear(c_f, K=0.5)

    a_V = np.full((network.Np, 1), fill_value=5., dtype=float)

    c = np.zeros((network.Np, Nc))
    c[:, dil] = np.linspace(0.1, 1., c.shape[0])
    mt = MulticomponentTools(network=network, num_components=Nc)

    x = c.reshape((-1, 1))
    dx = np.zeros_like(x)

    tol = 1e-12
    max_iter = 10
    ddt = mt.get_ddt(dt=1.)

    success = True
    x_old = x.copy()

    def ComputeSystem(x, c_l, type):
        J_ads, G_ads = AdsorptionSingleComponent(c=c_l,
                                                 dilute=dil, adsorbed=ads,
                                                 Vp=network['pore.volume'], a_v=a_V,
                                                 y_func=y_f, exclude=2,
                                                 type=type, dc=1e-6)
        G = ddt * (x - x_old) + G_ads
        if type == 'Defect':
            return G
        J = ddt + J_ads
        return J, G

    J, G = ComputeSystem(x, c, 'Jacobian')

    m_0 = c.copy()
    m_0[:, dil] *= network['pore.volume']
    m_0[:, ads] *= network['pore.volume'] * a_V.reshape((-1))
    for i in range(max_iter):
        last_iter = i
        dx = scipy.sparse.linalg.spsolve(J, -G).reshape(dx.shape)
        x = x + dx
        c = x.reshape(c.shape)
        G = ComputeSystem(x, c, 'Defect')
        G_norm = np.linalg.norm(np.abs(G), ord=2)
        if G_norm < tol:
            break
    if last_iter == max_iter - 1:
        print(f'WARNING: the maximum iterations ({max_iter}) were reached!')

    m = c.copy()
    m[:, dil] *= network['pore.volume']
    m[:, ads] *= network['pore.volume'] * a_V.reshape((-1))
    err = np.sum(m - m_0)/np.sum(m_0)
    success &= err < 1e-8

    theta_final = y_f(c[:, dil], c[:, ads])
    c_ads = theta_final
    err_ads = np.max(np.abs((c_ads-c[:, ads])/c_ads))
    success &= err_ads < 1e-5
    if output:
        print(f'{last_iter + 1} it [{G_norm:1.2e}]\
            mass-loss [{err:1.2e}] isotherm-error [{err_ads:1.2e}]')
    return success


def run_Langmuir(output: bool = True):
    Nx = 10
    Ny = 1
    Nz = 1
    Nc = 3
    spacing = 1./Nx

    # get network
    network = op.network.Cubic([Nx, Ny, Nz], spacing=spacing)

    # add geometry
    network.add_model_collection(op.models.collections.geometry.spheres_and_cylinders, domain='all')
    network.regenerate_models()

    # adsorption data
    ads = 0
    dil = 1

    ymax = np.full((network.Np, 1), fill_value=95., dtype=float)
    a_V = np.full((network.Np, 1), fill_value=10., dtype=float)

    def y_f(c_f, c_ads):
        return Langmuir(c_f.reshape((-1, 1)), K=0.1, y_max=ymax)

    c = np.zeros((network.Np, Nc))
    c[:, dil] = np.linspace(0.1, 1., c.shape[0])
    mt = MulticomponentTools(network=network, num_components=Nc)

    x = c.reshape((-1, 1))
    dx = np.zeros_like(x)

    tol = 1e-12
    max_iter = 100
    ddt = mt.get_ddt(dt=1.)

    success = True
    x_old = x.copy()

    def ComputeSystem(x, c_l, type):
        J_ads, G_ads = AdsorptionSingleComponent(c=c_l,
                                                 dilute=dil, adsorbed=ads,
                                                 Vp=network['pore.volume'], a_v=a_V,
                                                 y_func=y_f, exclude=2,
                                                 type=type, dc=1e-6)
        G = ddt * (x - x_old) + G_ads
        if type == 'Defect':
            return G
        J = ddt + J_ads
        return J, G

    J, G = ComputeSystem(x, c, 'Jacobian')

    m_0 = c.copy()
    m_0[:, dil] *= network['pore.volume']
    m_0[:, ads] *= network['pore.volume'] * a_V.reshape((-1))
    for i in range(max_iter):
        last_iter = i
        dx = scipy.sparse.linalg.spsolve(J, -G).reshape(dx.shape)
        x = x + dx
        c = x.reshape(c.shape)
        J, G = ComputeSystem(x, c, 'Jacobian')
        G_norm = np.linalg.norm(np.abs(G), ord=2)
        if G_norm < tol:
            break
    if last_iter == max_iter - 1:
        print(f'WARNING: the maximum iterations ({max_iter}) were reached!')

    m = c.copy()
    m[:, dil] *= network['pore.volume']
    m[:, ads] *= network['pore.volume'] * a_V.reshape((-1))
    err = np.sum(m - m_0)/np.sum(m_0)
    success &= err < 1e-12

    c_ads = y_f(c[:, dil], c[:, ads]).reshape((-1))
    err_ads = np.max(np.abs((c_ads-c[:, ads])/c_ads))
    success &= err_ads < 1e-5
    if output:
        print(f'{last_iter + 1} it [{G_norm:1.2e}]\
            mass-loss [{err:1.2e}] isotherm-error [{err_ads:1.2e}]')
    return success


def run_Freundlich(output: bool = True):
    Nx = 10
    Ny = 1
    Nz = 1
    Nc = 3
    spacing = 1./Nx

    # get network
    network = op.network.Cubic([Nx, Ny, Nz], spacing=spacing)

    # add geometry
    network.add_model_collection(op.models.collections.geometry.spheres_and_cylinders, domain='all')
    network.regenerate_models()

    # adsorption data
    ads = 0
    dil = 1

    a_V = np.full((network.Np, 1), fill_value=10., dtype=float)

    def y_f(c_f, c_ads):
        return Freundlich(c_f, 0.1, 1.5)

    c = np.zeros((network.Np, Nc))
    c[:, dil] = np.linspace(0.1, 1., c.shape[0])
    mt = MulticomponentTools(network=network, num_components=Nc)

    x = c.reshape((-1, 1))
    dx = np.zeros_like(x)

    tol = 1e-12
    max_iter = 100
    ddt = mt.get_ddt(dt=1.)

    success = True
    x_old = x.copy()

    def ComputeSystem(x, c_l, type):
        J_ads, G_ads = AdsorptionSingleComponent(c=c_l,
                                                 dilute=dil, adsorbed=ads,
                                                 Vp=network['pore.volume'], a_v=a_V,
                                                 y_func=y_f, exclude=2,
                                                 type=type, dc=1e-6)
        G = ddt * (x - x_old) + G_ads
        if type == 'Defect':
            return G
        J = ddt + J_ads
        return J, G

    J, G = ComputeSystem(x, c, 'Jacobian')

    m_0 = c.copy()
    m_0[:, dil] *= network['pore.volume']
    m_0[:, ads] *= network['pore.volume'] * a_V.reshape((-1))
    for i in range(max_iter):
        last_iter = i
        dx = scipy.sparse.linalg.spsolve(J, -G).reshape(dx.shape)
        x = x + dx
        c = x.reshape(c.shape)
        J, G = ComputeSystem(x, c, 'Jacobian')
        G_norm = np.linalg.norm(np.abs(G), ord=2)
        if G_norm < tol:
            break
    if last_iter == max_iter - 1:
        print(f'WARNING: the maximum iterations ({max_iter}) were reached!')

    m = c.copy()
    m[:, dil] *= network['pore.volume']
    m[:, ads] *= network['pore.volume'] * a_V.reshape((-1))
    err = np.sum(m - m_0)/np.sum(m_0)
    success &= err < 1e-12

    c_ads = y_f(c[:, dil], c[:, ads])
    err_ads = np.max(np.abs((c_ads-c[:, ads])/c_ads))
    success &= err_ads < 1e-5
    if output:
        print(f'{last_iter + 1} it [{G_norm:1.2e}]\
            mass-loss [{err:1.2e}] isotherm-error [{err_ads:1.2e}]')
    return success


if __name__ == '__main__':
    run()
