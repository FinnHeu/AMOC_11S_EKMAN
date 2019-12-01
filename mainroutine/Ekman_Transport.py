# EKMAN_TRANSPORT.PY

def Ekman_Transport(U: np.ndarray = np.ndarray,
                    V: np.ndarray = np.ndarray,
                    LAT: np.ndarray = np.ndarray
                    drag_coeff = 0,
                    extend_ranges = False,
                    ekman_layer_constant = True) -> np.ndarray

    # 0. Initialize functions
    %run def_check_and_reshape.py 
    %run def_bulkformulas_no_ext_range.py
    %run def_bulkformulas_ext_range.py
    %run def_windstress.py
    %run def_ekman_layer_depth.py # <--- not existing yet
    %run def_ekman_transp.py ??? # <--- not existing yet

    # 1. Check and reshape Data
    U_vec, V_vec = def_check_and_reshape(U,V)

    # 2. Compute wind stress
        # - NCEP
        # -	Large and Pond, 1981
        # -	Trenberth et al., 1990
        # -	Yelland and Taylor, 1996
        # -	Large and Yeager, 2004
    if not extend_ranges:
        %run bulkformulas_no_ext_range.py #define bulkformulas
        # run each bulk formula here
    elif extend_ranges:
        %run bulkformulas_ext_range.py #define bulkformulas
        # run each bulk formula here

    # 3. Compute varying Ekman Layer Depth
    if not ekman_layer_constant:
        %run
    elif ekman_layer_constant:
        %run

    # 4. Compute Ekman Transport







return(Psi, Cd)
