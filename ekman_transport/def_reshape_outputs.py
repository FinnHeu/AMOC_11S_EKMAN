# reshape_outputs.py

import numpy as np

def reshape_outputs(tau: np.ndarray=np.ndarray,
                    ekman_layer_depth: np.ndarray=np.ndarray,
                    f_coriolis: np.ndarray=np.ndarray) -> np.ndarray:

    ekman_layer_depth_array = np.empty_like(tau)
    f_coriolis_array = np.empty_like(tau)

    for i in range(5):
        ekman_layer_depth_array[:,i] = ekman_layer_depth
        f_coriolis_array[:,i] = f_coriolis

    return(ekman_layer_depth_array, f_coriolis_array)
