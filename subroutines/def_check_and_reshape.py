# Check and reshape Data
import numpy as np

def check_and_reshape(U: np.ndarray = np.ndarray, V: np.ndarray = np.ndarray) -> np.ndarray:

        if not np.shape(U) == np.shape(V):
            print('Data dimensions not consitent: Check dimensions')
        else:
            U_vec = np.concatenate(U)
            V_vec = np.concatenate(V)

            return(U_vec, V_vec)
