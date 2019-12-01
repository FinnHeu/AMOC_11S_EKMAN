# Check and reshape Data
import numpy as np

def check_and_reshape(U: np.ndarray = np.ndarray,
                      V: np.ndarray = np.ndarray,
                      LAT: np.ndarray = np.ndarray) -> np.ndarray:

        if not np.shape(U) == np.shape(V) == np.shape(LAT):
            print('Data dimensions not consitent: Check dimensions')
        else:
            U_vec = np.concatenate(U,axis=None)
            V_vec = np.concatenate(V,axis=None)
            LAT_vec = np.concatenate(LAT,axis=None)

            return(U_vec, V_vec, LAT_vec)
