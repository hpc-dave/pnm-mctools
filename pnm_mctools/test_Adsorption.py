import openpnm as op
import scipy.linalg
import scipy.sparse
import numpy as np
import scipy
from ToolSet import MulticomponentTools
from Adsorption import AdsorptionSingleComponent

def run(output: bool = True):
    success = True
    success &= run_Linear(output)
    success &= run_Langmuir(output)
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

    def ftheta(c_f, c_ads):
        return c_f*0.5

    ymax = np.full((network.Np, 1), fill_value=99., dtype=float)
    a_V = np.full((network.Np, 1), fill_value=5., dtype=float)

    c = np.zeros((network.Np, Nc))
    c[:, dil] = 1.
    mt = MulticomponentTools(network=network, num_components=Nc)

    x = c.reshape((-1, 1))
    dx = np.zeros_like(x)

    tol = 1e-12
    max_iter = 10
    ddt = mt.DDT(dt=1.)

    success = True
    x_old = x.copy()

    def ComputeSystem(x, c_l, type):
        J_ads, G_ads = AdsorptionSingleComponent(c=c_l,
                                                 dilute=dil, adsorbed=ads,
                                                 y_max=ymax, Vp=network['pore.volume'],
                                                 a_v=a_V, ftheta=ftheta, exclude=2,
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
    success &= err < 1e-12

    theta_final = ftheta(c[:, dil], c[:, ads])
    c_ads = ymax * theta_final
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

    def ftheta(c_f, c_ads):
        Kcf = c_f
        return Kcf / (1+Kcf)

    ymax = np.full((network.Np, 1), fill_value=95., dtype=float)
    a_V = np.full((network.Np, 1), fill_value=10., dtype=float)

    c = np.zeros((network.Np, Nc))
    c[:, dil] = 1.
    mt = MulticomponentTools(network=network, num_components=Nc)

    x = c.reshape((-1, 1))
    dx = np.zeros_like(x)

    tol = 1e-12
    max_iter = 100
    ddt = mt.DDT(dt=1.)

    success = True
    x_old = x.copy()

    def ComputeSystem(x, c_l, type):
        J_ads, G_ads = AdsorptionSingleComponent(c=c_l,
                                                 dilute=dil, adsorbed=ads,
                                                 y_max=ymax, Vp=network['pore.volume'],
                                                 a_v=a_V, ftheta=ftheta, exclude=2,
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

    theta_final = ftheta(c[:, dil], c[:, ads])
    c_ads = ymax * theta_final
    err_ads = np.max(np.abs((c_ads-c[:, ads])/c_ads))
    success &= err_ads < 1e-5
    if output:
        print(f'{last_iter + 1} it [{G_norm:1.2e}]\
            mass-loss [{err:1.2e}] isotherm-error [{err_ads:1.2e}]')
    return success
