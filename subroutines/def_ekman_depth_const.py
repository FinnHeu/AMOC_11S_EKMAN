# ekman_depth_const.py

def ekman_depth_const(U: np.ndarray=np.ndarray,
                      V: np.ndarray=np.ndarray,
                      LAT: np.ndarray=np.ndarray) -> np.ndarray:

      standard_depth = 100 #m
      ekman_layer_depth = np.ones_like(U) * standard_depth

      return ekman_layer_depth
