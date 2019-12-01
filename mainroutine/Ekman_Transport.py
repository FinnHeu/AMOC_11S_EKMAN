# EKMAN_TRANSPORT.PY

def Ekman_Transport(U: np.ndarray = np.ndarray,
                    V: np.ndarray = np.ndarray,
                    LAT: np.ndarray = np.ndarray
                    drag_coeff = 0,
                    extend_ranges = False,
                    ekman_layer_constant = True) -> np.ndarray

    # 1. Check and reshape Data
        # 1.1 Shape of U, V, LAT, LON
        # 1.2 Make Vektor out of data arrays

    # 2. Compute wind stress
        # - NCEP
        # -	Large and Pond, 1981
        # -	Trenberth et al., 1990
        # -	Yelland and Taylor, 1996
        # -	Large and Yeager, 2004
    if not extend_ranges:
        %run bulkformulas_no_ext_range.py
    elif extend_ranges:
        %run bulkformulas_ext_range.py

    # 3. Compute varying Ekman Layer Depth
    if not ekman_layer_constant:
        %run
    elif ekman_layer_constant:
        %run

    # 4. Compute Ekman Transport







return(Psi, Cd)
