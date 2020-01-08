# drag_coefficients.py
import numpy as np


def drag_coefficients(U: np.ndarray = np.ndarray) -> np.ndarray:

    # input: U: absolute wind speed UV_abs

    Cd = np.empty([U.shape[0], 5])

    Cd[:, 0] = ncep_ncar_2007(U)
    Cd[:, 1] = large_and_pond_1981(U)
    Cd[:, 2] = trenberth_etal_1990(U)
    Cd[:, 3] = yelland_and_taylor_1996(U)
    Cd[:, 4] = large_and_yeager_2004(U)

    return Cd


def drag_coefficients_ext(U: np.ndarray = np.ndarray) -> np.ndarray:

    # input: U: absolute wind speed UV_abs

    Cd = np.empty([U.shape[0], 5])

    Cd[:, 0] = ncep_ncar_2007_ext(U)
    Cd[:, 1] = large_and_pond_1981_ext(U)
    Cd[:, 2] = trenberth_etal_1990_ext(U)
    Cd[:, 3] = yelland_and_taylor_1996_ext(U)
    Cd[:, 4] = large_and_yeager_2004_ext(U)

    return Cd
