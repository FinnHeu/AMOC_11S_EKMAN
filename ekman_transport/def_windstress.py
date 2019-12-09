# windstress.py

import numpy as np

def windstress(U: np.ndarray=np.ndarray,Cd: np.ndarray=np.ndarray) -> np.ndarray:

    # input
    # U: absolute wind speed UV_abs
    # Cd: drag coefficient of absolute wind speed

    # Set parameters
    rho_air = np.ones_like(U,dtype='float64') * 1.2041 #kg/m³

    tau = np.ones_like(Cd,dtype='float64')

    # Calculate windstress
    for i in range(5):
        tau[:,i] = rho_air * Cd[:,i] * U**2

    return(tau)
