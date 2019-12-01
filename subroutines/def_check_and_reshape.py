# Check and reshape Data
import numpy as np

def check_and_reshape(U: np.ndarray = np.ndarray,
                      V: np.ndarray = np.ndarray,
                      LAT: np.ndarray = np.ndarray) -> np.ndarray:

        if not np.shape(U) == np.shape(V):
            print('Data dimensions not consitent: Check dimensions')
        else:
            U_vec = np.concatenate(U)
            V_vec = np.concatenate(V)

            n = len(U_vec) / len(LAT)
            LAT_vec = np.tile(np.concatenate(LAT),n)


            return(U_vec, V_vec)
