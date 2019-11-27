# EKMAN_TRANSPORT.PY

def Ekman_Transport(U: np.ndarray = np.ndarray,
                    V: np.ndarray = np.ndarray,
                    LAT: np.ndarray = np.ndarray
                    drag_coeff = 'all',
                    extend_ranges = False,
                    ekman_layer_constant = True) -> np.ndarray

# 1. Check Data
    # 1.1 Shape of U, V, LAT, LON

# 2. Compute wind stress
    # - NCEP
    # -	Large and Pond, 1981
    # -	Trenberth et al., 1990
    # -	Yelland and Taylor, 1996
    # -	Large and Yeager, 2004
bulkformulas.py
# 3. Compute varying Ekman Layer Depth


# 4. Compute Ekman Transport







return(Psi, Cd)
