# rotateion_matrix.py

import numpy as np

def rot_matrix(alpha: np.ndarray=np.ndarray) -> np.ndarray:

    T = np.empty([2,2])
    T[0,0] = np.cos(alpha)
    T[0,1] = -np.sin(alpha)
    T[1,0] = np.sin(alpha)
    T[1,1] = np.cos(alpha)

    return T
