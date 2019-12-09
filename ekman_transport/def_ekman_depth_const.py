# ekman_depth_const.py

import numpy as np

def ekman_depth_const(U: np.ndarray=np.ndarray) -> np.ndarray:

    # INPUT
    # U: absoute wind speed UV_abs    
    standard_depth = 100 #m
    ekman_layer_depth = np.ones_like(U) * standard_depth

    return ekman_layer_depth
