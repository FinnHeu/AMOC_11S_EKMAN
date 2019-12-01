# EKMAN_TRANSPORT.PY

def Ekman_Transport(U: np.ndarray = np.ndarray,
                    V: np.ndarray = np.ndarray,
                    LAT: np.ndarray = np.ndarray
                    drag_coeff = 0,
                    extend_ranges = False,
                    ekman_layer_constant = True) -> np.ndarray

# INPUT:
# U:                        [n x m] zonal wind speed
# V:                        [n x m] merdidional wind speed
# LAT:                      [n x m] latitude
# drag_coeff:               0|1|2|3|4|
# extend_ranges:            True | False
# ekman_layer_constant:     True | False
#
#


    # 0. Initialize functions
    %run def_check_and_reshape.py

    if not extend_ranges:
        %run def_bulkformulas_no_ext_range.py
    else:
        %run def_bulkformulas_ext_range.py

    %run def_drag_coefficients.py
    %run def_windstress.py
    %run def_ekman_depth_const.py
    %run def_ekman_depth_vary.py 

    # 1. Check and reshape Data
    U_vec, V_vec, LAT_vec = check_and_reshape(U,V,LAT) # Reshapes all input data into vectors

    # 2. Compute Drag Coefficients
    Cd_U, Cd_V = drag_coefficients(U=U_vec, V=V_vec)

    # 3. Compute Wind Stress
    taux, tauy = windstress(U=U_vec, V=V_vec, Cd_U=Cd_U, Cd_V=Cd_V)

    # 3. Compute varying Ekman Layer Depth
    if not ekman_layer_constant:
        %run
    elif ekman_layer_constant:
        %run

    # 4. Compute Ekman Transport







return(Psi, Cd)
