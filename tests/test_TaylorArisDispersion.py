from math import pi
from math import sqrt
import openpnm as op
import pnm_mctools.models.const_spheres_and_cylinders as geo_model     # noqa: E402
import numpy as np                                                     # noqa: E402
from pnm_mctools.ToolSet import MulticomponentTools                    # noqa: E402
from pnm_mctools.TaylorArisDispersion import Coefficient as TACoef


def test_TaylorArisDispersion(output=False):
    Nx = 10
    Ny = 1
    Nz = 1
    Nc = 1
    spacing = 1./Nx
    Dbin = 1.

    # get network
    network = op.network.Cubic([Nx, Ny, Nz], spacing=spacing)

    # add geometry
    geo = geo_model.spheres_and_cylinders
    network.add_model_collection(geo, domain='all')
    network.regenerate_models()
    network['throat.length'] = network['pore.coords'][1:, 0] - network['pore.coords'][:-1, 0]
    network['throat.radius'] = network['throat.diameter']*0.5
    t_radius = network['throat.radius'][0]
    assert np.all(network['throat.radius'] == t_radius), 'The radii need to be equal everywhere for this test'

    Nt = network.num_throats()
    mt = MulticomponentTools(network=network, num_components=Nc)

    rate = 1.
    rates = np.full_like(network['throat.length'], fill_value=rate, dtype=float)

    # the minimal dispersion coefficient by varying the binary diffusion coefficient
    # can be derived from Taylor Aris dispersion by dD_Ta/dD_bin = 1- Q_h/(48*pi^2*r^2*Dbin^2)=0
    Dbin_min = rate/(sqrt(48)*pi*t_radius)
    min_pos = int(Nt/2)

    Dbin = np.arange(0.5, 1.5, 1./Nt) * Dbin_min
    Dbin[min_pos] = Dbin_min

    D_ta = TACoef(mt, rate=rates, Dbin=Dbin, throat_radius='throat.radius')
    D_ta_arr = TACoef(network, rate=rates, Dbin=Dbin, throat_radius=network['throat.radius'])

    assert np.all(D_ta == D_ta_arr), 'Both arrays need to be exactly the same'
    assert np.argmin(D_ta) == min_pos, 'The position of the minimum is off'


if __name__ == "__main__":
    test_TaylorArisDispersion(True)
