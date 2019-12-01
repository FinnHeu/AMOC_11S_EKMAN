# drag_coefficients.py

def drag_coefficients(U: np.ndarray, V: np.ndarray) -> np.ndarray:

    Cd_U = np.empty([U_vec.shape[0],5])
    Cd_V = np.empty([V_vec.shape[0],5])

    Cd_U[:,0] = ncep_ncar_2007(U_vec)
    Cd_U[:,1] = large_and_pond_1981(U_vec)
    Cd_U[:,2] = trenberth_etal_1990(U_vec)
    Cd_U[:,3] = yelland_and_taylor_1996(U_vec)
    Cd_U[:,4] = large_and_yeager_2004(U_vec)

    Cd_V[:,0] = ncep_ncar_2007(V_vec)
    Cd_V[:,1] = large_and_pond_1981(V_vec)
    Cd_V[:,2] = trenberth_etal_1990(V_vec)
    Cd_V[:,3] = yelland_and_taylor_1996(V_vec)
    Cd_V[:,4] = large_and_yeager_2004(V_vec)

    return(Cd_U, Cd_V)
