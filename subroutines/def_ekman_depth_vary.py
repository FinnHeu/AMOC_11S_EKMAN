# ekman_depth_vary.py

def ekman_depth_vary(U: np.ndarray=np.ndarray, V:np.ndarray=np.ndarray, LAT: np.ndarray=np.ndarray) -> np.ndarray:

    total_wind_speed = np.empty_like(U)
    total_wind_speed = np.sqrt(U**2 + V**2)

    ekman_layer_depth = np.empty_like(U)
    ekman_layer_depth = np.round((total_wind_speed * 7.6) /
                        np.sqrt(np.sin(np.abs(np.deg2rad(LAT)))),decimals=0)

    return ekman_layer_depth
