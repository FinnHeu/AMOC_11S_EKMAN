# ekman_transport_calc.py

import numpy as np

def ekman_transport_calc(tau: np.ndarray=np.ndarray, ekman_layer_depth_array: np.ndarray=np.ndarray, f_coriolis_array: np.ndarray=np.ndarray,alpha: np.ndarray=np.ndarray) -> np.ndarray:

    # set parameters
    rho = 1028 #kg/m³

    # compute V_0
    V_0 = (np.sqrt(2) * np.pi * tau / (ekman_layer_depth_array * rho * abs(f_coriolis_array)));

    ekman_transp_u = np.empty_like(V_0)
    ekman_transp_v = np.empty_like(V_0)

    for i in range(V_0.shape[0]):
        for j in range(V_0.shape[1]):

            z = np.arange(-ekman_layer_depth_array[i,j],1)
            ekman_vel_u = np.empty_like(z)
            ekman_vel_v = np.empty_like(z)

            for k in range(len(z)):

                ekman_vel_u[k] = -V_0[i,j] * np.sin(np.pi + (np.pi/4) + (np.pi/ekman_layer_depth_array[i,j]) * z[k]) * np.exp((np.pi/ekman_layer_depth_array[i,j]) * z[k])
                ekman_vel_v[k] = V_0[i,j] * np.cos((np.pi/4) + (np.pi/ekman_layer_depth_array[i,j]) * z[k]) * np.exp((np.pi/ekman_layer_depth_array[i,j]) * z[k])

                ekman_transp_u[i,j] = np.trapz(ekman_vel_u,axis=0)
                ekman_transp_v[i,j] = np.trapz(ekman_vel_v,axis=0)

    return(ekman_transp_u, ekman_transp_v)

        
