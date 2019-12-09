# ekman_depth_vary.py

import numpy as np

def ekman_depth_vary(U: np.ndarray=np.ndarray, LAT: np.ndarray=np.ndarray) -> np.ndarray:
    # input
    # U: absolute wind speed UV_abs
    # LAT: latitude in degree
    ekman_layer_depth = np.empty_like(U)
    ekman_layer_depth = np.round((U * 7.6) /
                        np.sqrt(np.sin(np.abs(np.deg2rad(LAT)))),decimals=0)

    return(ekman_layer_depth)
