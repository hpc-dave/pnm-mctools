from math import pi
from math import sqrt
import openpnm as op
import pnm_mctools.models.const_spheres_and_cylinders as geo_model     # noqa: E402
import numpy as np                                                     # noqa: E402
from pnm_mctools.ToolSet import MulticomponentTools                    # noqa: E402
from pnm_mctools.Miscellaneous import compute_pore_residence_time


def test_PoreResidenceTime(output=False):

    Nx = 10
    Ny = 1
    Nz = 1
    spacing = 1./Nx

    # get network
    network = op.network.Cubic([Nx, Ny, Nz], spacing=spacing)

    Np = network.num_pores()

    # add geometry
    geo = geo_model.spheres_and_cylinders
    network.add_model_collection(geo, domain='all')
    network.regenerate_models()
    rate = np.ones_like(network['throat.diameter'], dtype=float)
    network['pore.volume'] *= np.arange(0.5, 1.5, 1./Np)
    tau_inflow = compute_pore_residence_time(Q=rate, network=network, approach='inflow')
    tau_outflow = compute_pore_residence_time(Q=rate, network=network, approach='outflow')
    tau_min = compute_pore_residence_time(Q=rate, network=network, approach='min')

    tau_min_comp = np.min(np.hstack([tau_inflow, tau_outflow]), axis=1).reshape((-1, 1))
    assert np.all(tau_min_comp - tau_min < 1e-14), 'inconsistent values'

    tau_ex = network['pore.volume'].reshape((-1, 1))/1.

    assert np.all(np.abs(tau_ex - tau_min) < 1e-14), 'value difference exceeds tolerance'


if __name__ == "__main__":
    test_PoreResidenceTime(True)
