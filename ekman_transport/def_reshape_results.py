# reshape_results.py

import numpy as np

def reshape_results(U: np.ndarray=np.ndarray, U_ek_int: np.ndarray=np.ndarray, V_ek_int: np.ndarray=np.ndarray, LAT_vec: np.ndarray=np.ndarray) -> np.ndarray:
    # input:
    # U: original array
    a = np.shape(U)

    if len(a) == 1:
        U_ek_int_reshaped = U_ek_int
        V_ek_int_reshaped = V_ek_int
        LAT_reshaped = LAT_vec

    elif len(a) == 2:

        U_ek_int_reshaped = np.empty([a[0],a[1],5])
        V_ek_int_reshaped = np.empty([a[0],a[1],5])
        LAT_reshaped = LAT_vec.reshape(a)

        for i in range(5):
            U_ek_int_reshaped[:,:,i] = U_ek_int[:,i].reshape(a)
            V_ek_int_reshaped[:,:,i] = V_ek_int[:,i].reshape(a)

    elif len(a) == 3:

        U_ek_int_reshaped = np.empty([a[0],a[1],a[2],5])
        V_ek_int_reshaped = np.empty([a[0],a[1],5])
        LAT_reshaped = LAT_vec.reshape(a)

        for i in range(5):
            U_ek_int_reshaped[:,:,:,i] = U_ek_int[:,i].reshape(a)
            V_ek_int_reshaped[:,:,:,i] = V_ek_int[:,i].reshape(a)



    return U_ek_int_reshaped, V_ek_int_reshaped, LAT_reshaped
