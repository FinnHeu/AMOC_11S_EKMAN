# rotate_vector.py

import numpy as np

def rotate_vector(U_ek_int: np.ndarray = np.ndarray, V_ek_int: np.ndarray = np.ndarray, alpha: np.ndarray = np.ndarray) -> np.ndarray:

    real_u_ek = np.empty_like(U_ek_int)
    real_v_ek = np.empty_like(U_ek_int)

    for i in range(len(alpha)):
        for j in range(5):

            T = rot_matrix(alpha[i])

            vel = np.empty([2,1])
            vel[0,0] = U_ek_int[i,j]
            vel[1,0] = V_ek_int[i,j]

            real_u_ek[i,j] = np.matmul(T,vel)[0]
            real_v_ek[i,j] = np.matmul(T,vel)[1]

    return real_u_ek, real_v_ek
