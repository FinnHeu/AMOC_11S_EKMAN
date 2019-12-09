# coriolis parameter

import numpy as np

def coriolis_parameter(LAT: np.ndarray=np.ndarray) -> np.ndarray:

    omega = 2 * np.pi/(3600*24)
    f_coriolis = 2 * omega * np.sin(np.deg2rad(LAT))

    return f_coriolis
