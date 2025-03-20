import numpy as np
from .Miscellaneous import compute_throat_peclet_number


def Coefficient(c, rate: np.ndarray, Dbin: np.ndarray | float, throat_radius: str | np.ndarray) -> np.ndarray:
    Pe_loc = compute_throat_peclet_number(c=c, Q=rate, Dbin=Dbin, throat_radius=throat_radius)
    D_TA = Dbin.reshape(Pe_loc.shape) * (1 + Pe_loc**2 / 48.)
    return D_TA
