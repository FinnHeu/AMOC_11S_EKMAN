# angle to pure east

import numpy as np


def angle_to_pure_east(
    U: np.ndarray = np.ndarray, V: np.ndarray = np.ndarray
) -> np.ndarray:

    real_wind = np.array([U, V])
    assumed_wind = np.array([1, 0])

    # use dot product to compute angle between purely east and real wind direction
    alpha = np.empty_like(U, dtype="float64")
    for i in range(len(U)):
        if np.logical_and(U[i] == 0, V[i] == 0):
            alpha[i] = np.nan
        else:
            alpha[i] = np.arccos(
                (real_wind[0, i] * assumed_wind[0] + real_wind[1, i] * assumed_wind[1])
                / (
                    np.sqrt(real_wind[0, i] ** 2 + real_wind[1, i] ** 2)
                    * np.sqrt(assumed_wind[0] ** 2 + assumed_wind[1] ** 2)
                )
            )

    return alpha
