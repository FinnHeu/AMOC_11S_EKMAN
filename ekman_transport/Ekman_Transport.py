# EKMAN_TRANSPORT.PY
import numpy as np

def Ekman_Transport(U: np.ndarray = np.ndarray,
                    V: np.ndarray = np.ndarray,
                    LAT: np.ndarray = np.ndarray,
                    drag_coeff = 5,
                    extend_ranges = False,
                    ekman_layer_constant = True) -> np.ndarray:

    ''' 
    EKMAN_TRANSPORT.PY returns the zonal and meridional Ekman transport [m²/2]
    from a given windfield, by vertically integrating the Ekman spiral 
    at each location from the surface to either 100m (ekman_layer_const=True)
    or to a wind dependent ekman depth (ekman_layer_const=False). Several 
    drag coefficients can be chosen either extrapolating (extend_ranges=True) 
    or masking (extend_ranges=False) in non defined ranges.
    
    # INPUT:
    # U:                        [n x m] zonal wind speed
    # V:                        [n x m] merdidional wind speed
    # LAT:                      [n x m] latitude
    # drag_coeff:               0 = ncep_ncar_2007 | 1 = large_and_pond_1981 | 2 = trenberth_et_al_1990 |
    #                           3 = yelland_and_taylor_1996 | 4 = large_and_yeager_2006 | 5 = all
    # extend_ranges:            True | False
    # ekman_layer_constant:     True | False
    #
    #
    # OUTPUT:
    # U_ek:                     Zonal ekman velocity, vertically integrated
    # V_ek:                     Meridional ekman velocity, vertically integrated 
    '''



    # 0. Initialize functions
    
    
    


    # 1. Check and reshape Data
    U_vec, V_vec, LAT_vec, UV_abs = check_and_reshape(U=U,V=V,LAT=LAT) # Reshapes all input data into vectors

    # 2. Compute Drag Coefficients
    if not extend_ranges:
        Cd = drag_coefficients(U=UV_abs)
    elif extend_ranges:
        Cd = drag_coefficients_ext(U=UV_abs)

    # 3. Compute Wind Stress
    tau = windstress(U=UV_abs, Cd=Cd)

    # 3. Compute Ekman Layer Depth
    if not ekman_layer_constant:
        ekman_layer_depth = ekman_depth_const(U=UV_abs)
    else:
        ekman_layer_depth = ekman_depth_vary(U=UV_abs, LAT=LAT_vec)

    # 4. Compute Coriolis paramter
    f_coriolis = coriolis_parameter(LAT_vec)

    # 5. Reshape Outputs
    ekman_layer_depth_array, f_coriolis_array = reshape_outputs(tau=tau,ekman_layer_depth=ekman_layer_depth,f_coriolis=f_coriolis)

    # 6. Compute angle to purely east
    alpha = angle_to_pure_east(U=U_vec,V=V_vec)

    # 7. Compute vertically integrated Ekman transport assuming wind is blowing only in positive u direction
    ekman_transp_u, ekman_transp_v = ekman_transport_calc(tau=tau,ekman_layer_depth_array=ekman_layer_depth_array,f_coriolis=f_coriolis_array,alpha=alpha)

    # 8. Rotate Ekman transport to true direction
    real_u_ek, real_v_ek = rotate_vector(U_ek_int=ekman_transp_u,V_ek_int=ekman_transp_v,alpha=alpha) # m²/s

    # 9. Reshape data into original shape
    U_ek_int_reshaped, V_ek_int_reshaped, LAT_reshaped = reshape_results(U=U,U_ek_int=real_u_ek,V_ek_int=real_v_ek,LAT_vec=LAT_vec)

    # 10. Rename data
    U_ekman = U_ek_int_reshaped
    V_ekman = V_ek_int_reshaped
    
    # 11. Give Output
          
    if drag_coeff == 1:
        print('Drag Coefficient: ncep_ncar_2007')
        return(U_ekman[:,:,0], V_ekman[:,:,0])
        
    elif drag_coeff == 2:
        print('Drag Coefficient: large_and_pond_1981')
        return(U_ekman[:,:,1], V_ekman[:,:,1])
        
    elif drag_coeff == 3:
        print('Drag Coefficient: trenberth_etal_1990')
        return(U_ekman[:,:,0], V_ekman[:,:,0])
        
    elif drag_coeff == 3:
        print('Drag Coefficient: yelland_and_taylor_1996')
        return(U_ekman[:,:,3], V_ekman[:,:,3])
        
    elif drag_coeff == 4:
        print('Drag Coefficient: large_and_yeager_2004')
        return(U_ekman[:,:,4], V_ekman[:,:,4])
        
    elif drag_coeff == 5:
        print('Drag Coefficient: [:,:,0] ncep_ncar_2007')
        print('Drag Coefficient: [:,:,1] large_and_pond_1981')
        print('Drag Coefficient: [:,:,2] trenberth_etal_1990')
        print('Drag Coefficient: [:,:,3] yelland_and_taylor_1996')
        print('Drag Coefficient: [:,:,4] large_and_yeager_2004')

        return(U_ekman[:,:,:], V_ekman[:,:,:])
        