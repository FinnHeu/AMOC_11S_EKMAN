# Check and reshape Data
import numpy as np

def check_and_reshape(U: np.ndarray = np.ndarray,
                      V: np.ndarray = np.ndarray,
                      LAT: np.ndarray = np.ndarray) -> np.ndarray:
    # returns 

    if not np.shape(U) == np.shape(V) == np.shape(LAT):
        print('Data dimensions not consitent: Check dimensions')
    else:
        U_vec = np.concatenate(U,axis=None)
        V_vec = np.concatenate(V,axis=None)
        LAT_vec = np.concatenate(LAT,axis=None)
        UV_abs = np.sqrt(U_vec**2 + V_vec**2)

        return(U_vec, V_vec, LAT_vec, UV_abs)
