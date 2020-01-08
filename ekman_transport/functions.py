# all_functions.py

# angle to pure east

import numpy as np


def angle_to_pure_east(
    U: np.ndarray = np.ndarray, V: np.ndarray = np.ndarray
) -> np.ndarray:

    real_wind = np.array([U, V])
    assumed_wind = np.array([1, 0])

    # use dot product to compute angle between purely east and real wind direction
    alpha = np.empty_like(U, dtype="float64")
    for i in range(len(U)):
        if np.logical_and(U[i] == 0, V[i] == 0):
            alpha[i] = np.nan
        else:
            alpha[i] = np.arccos(
                (real_wind[0, i] * assumed_wind[0] + real_wind[1, i] * assumed_wind[1])
                / (
                    np.sqrt(real_wind[0, i] ** 2 + real_wind[1, i] ** 2)
                    * np.sqrt(assumed_wind[0] ** 2 + assumed_wind[1] ** 2)
                )
            )

    return alpha


# Check and reshape Data
import numpy as np


def check_and_reshape(
    U: np.ndarray = np.ndarray, V: np.ndarray = np.ndarray, LAT: np.ndarray = np.ndarray
) -> np.ndarray:
    # returns

    if not np.shape(U) == np.shape(V) == np.shape(LAT):
        print("Data dimensions not consitent: Check dimensions")
    else:
        U_vec = np.concatenate(U, axis=None)
        V_vec = np.concatenate(V, axis=None)
        LAT_vec = np.concatenate(LAT, axis=None)
        UV_abs = np.sqrt(U_vec ** 2 + V_vec ** 2)

        return (U_vec, V_vec, LAT_vec, UV_abs)


# coriolis parameter

import numpy as np


def coriolis_parameter(LAT: np.ndarray = np.ndarray) -> np.ndarray:

    omega = 2 * np.pi / (3600 * 24)
    f_coriolis = 2 * omega * np.sin(np.deg2rad(LAT))

    return f_coriolis


# drag_coefficients.py
import numpy as np


def drag_coefficients(U: np.ndarray = np.ndarray) -> np.ndarray:

    # input: U: absolute wind speed UV_abs

    Cd = np.empty([U.shape[0], 5])

    Cd[:, 0] = ncep_ncar_2007(U)
    Cd[:, 1] = large_and_pond_1981(U)
    Cd[:, 2] = trenberth_etal_1990(U)
    Cd[:, 3] = yelland_and_taylor_1996(U)
    Cd[:, 4] = large_and_yeager_2004(U)

    return Cd


def drag_coefficients_ext(U: np.ndarray = np.ndarray) -> np.ndarray:

    # input: U: absolute wind speed UV_abs

    Cd = np.empty([U.shape[0], 5])

    Cd[:, 0] = ncep_ncar_2007_ext(U)
    Cd[:, 1] = large_and_pond_1981_ext(U)
    Cd[:, 2] = trenberth_etal_1990_ext(U)
    Cd[:, 3] = yelland_and_taylor_1996_ext(U)
    Cd[:, 4] = large_and_yeager_2004_ext(U)

    return Cd


# ekman_depth_const.py

import numpy as np


def ekman_depth_const(U: np.ndarray = np.ndarray) -> np.ndarray:

    # INPUT
    # U: absoute wind speed UV_abs
    standard_depth = 100  # m
    ekman_layer_depth = np.ones_like(U) * standard_depth

    return ekman_layer_depth


# ekman_depth_vary.py

import numpy as np


def ekman_depth_vary(
    U: np.ndarray = np.ndarray, LAT: np.ndarray = np.ndarray
) -> np.ndarray:
    # input
    # U: absolute wind speed UV_abs
    # LAT: latitude in degree
    ekman_layer_depth = np.empty_like(U)
    ekman_layer_depth = np.round(
        (U * 7.6) / np.sqrt(np.sin(np.abs(np.deg2rad(LAT)))), decimals=0
    )

    return ekman_layer_depth


# ekman_transport_calc.py

import numpy as np


def ekman_transport_calc(
    tau: np.ndarray = np.ndarray,
    ekman_layer_depth_array: np.ndarray = np.ndarray,
    f_coriolis_array: np.ndarray = np.ndarray,
    alpha: np.ndarray = np.ndarray,
) -> np.ndarray:

    # set parameters
    rho = 1028  # kg/m³

# replace zeros by dummy                                                                                    # <-------------------------
    dummy = 99999                                                                                           # <-------------------------
    ekman_layer_depth_array = np.where(ekman_layer_depth_array == False, dummy, ekman_layer_depth_array)    # <-------------------------

    # compute V_0
    V_0 = (
        np.sqrt(2)
        * np.pi
        * tau
        / (ekman_layer_depth_array * rho * abs(f_coriolis_array))
    )

    ekman_transp_u = np.empty_like(V_0)
    ekman_transp_v = np.empty_like(V_0)

    for i in range(V_0.shape[0]):
        for j in range(V_0.shape[1]):

            z = np.arange(-ekman_layer_depth_array[i, j], 1)
            ekman_vel_u = np.empty_like(z)
            ekman_vel_v = np.empty_like(z)

            for k in range(len(z)):
                if ekman_layer_depth_array[i,j] == dummy:                # <-------------------------
                    V_0[i,j] = 0                                             # <-------------------------

                ekman_vel_u[k] = (
                -V_0[i, j]
                * np.sin(
                    np.pi
                    + (np.pi / 4)
                    + (np.pi / ekman_layer_depth_array[i, j]) * z[k]
                )
                * np.exp((np.pi / ekman_layer_depth_array[i, j]) * z[k])
                )
                ekman_vel_v[k] = (
                V_0[i, j]
                * np.cos(
                    (np.pi / 4) + (np.pi / ekman_layer_depth_array[i, j]) * z[k]
                )
                * np.exp((np.pi / ekman_layer_depth_array[i, j]) * z[k])
                )

            ekman_transp_u[i, j] = np.trapz(ekman_vel_u, axis=0)
            ekman_transp_v[i, j] = np.trapz(ekman_vel_v, axis=0)

        # remove dummy transports for |U| == 0

    return (ekman_transp_u, ekman_transp_v)


# reshape_outputs.py

import numpy as np


def reshape_outputs(
    tau: np.ndarray = np.ndarray,
    ekman_layer_depth: np.ndarray = np.ndarray,
    f_coriolis: np.ndarray = np.ndarray,
) -> np.ndarray:

    ekman_layer_depth_array = np.empty_like(tau)
    f_coriolis_array = np.empty_like(tau)

    for i in range(5):
        ekman_layer_depth_array[:, i] = ekman_layer_depth
        f_coriolis_array[:, i] = f_coriolis

    #ekman_layer_depth_array = np.where(ekman_layer_depth_array == False, np.nan, ekman_layer_depth_array)

    return (ekman_layer_depth_array, f_coriolis_array)


# reshape_results.py

import numpy as np


def reshape_results(
    U: np.ndarray = np.ndarray,
    U_ek_int: np.ndarray = np.ndarray,
    V_ek_int: np.ndarray = np.ndarray,
    LAT_vec: np.ndarray = np.ndarray,
) -> np.ndarray:
    # input:
    # U: original array
    a = np.shape(U)

    if len(a) == 1:
        U_ek_int_reshaped = U_ek_int
        V_ek_int_reshaped = V_ek_int
        LAT_reshaped = LAT_vec

    elif len(a) == 2:

        U_ek_int_reshaped = np.empty([a[0], a[1], 5])
        V_ek_int_reshaped = np.empty([a[0], a[1], 5])
        LAT_reshaped = LAT_vec.reshape(a)

        for i in range(5):
            U_ek_int_reshaped[:, :, i] = U_ek_int[:, i].reshape(a)
            V_ek_int_reshaped[:, :, i] = V_ek_int[:, i].reshape(a)

    elif len(a) == 3:

        U_ek_int_reshaped = np.empty([a[0], a[1], a[2], 5])
        V_ek_int_reshaped = np.empty([a[0], a[1], a[2], 5])
        LAT_reshaped = LAT_vec.reshape(a)

        for i in range(5):
            U_ek_int_reshaped[:, :, :, i] = U_ek_int[:, i].reshape(a)
            V_ek_int_reshaped[:, :, :, i] = V_ek_int[:, i].reshape(a)

    return U_ek_int_reshaped, V_ek_int_reshaped, LAT_reshaped


# rotation_matrix.py

import numpy as np


def rot_matrix(alpha: np.ndarray = np.ndarray) -> np.ndarray:

    T = np.empty([2, 2])
    T[0, 0] = np.cos(alpha)
    T[0, 1] = -np.sin(alpha)
    T[1, 0] = np.sin(alpha)
    T[1, 1] = np.cos(alpha)

    return T

# add_mean.property.py
def add_mean(U: np.ndarray=np.ndarray, V: np.ndarray=np.ndarray) -> np.ndarray:

    import numpy as np

    a = U.shape

    if len(a) in np.arange(1,3,1):
        dummy_u = np.empty([a[0],a[1],6])
        dummy_u[:,:,5] = np.nanmean(U, axis=-1)
        dummy_u[:,:,np.arange(0,5,1)] = U
        U_out = dummy_u

        dummy_v = np.empty([a[0],a[1],6])
        dummy_v[:,:,5] = np.nanmean(V, axis=-1)
        dummy_v[:,:,np.arange(0,5,1)] = V
        V_out = dummy_v
    if len(a) == 3:
        dummy_u = np.empty([a[0],a[1],6])
        dummy_u[:,:,5] = np.nanmean(U, axis=-1)
        dummy_u[:,:,np.arange(0,5,1)] = U
        U_out = dummy_u

        dummy_v = np.empty([a[0],a[1],6])
        dummy_v[:,:,5] = np.nanmean(V, axis=-1)
        dummy_v[:,:,np.arange(0,5,1)] = V
        V_out = dummy_v

    return(U_out,V_out)

# rotate_vector.py

import numpy as np


def rotate_vector(
    U_ek_int: np.ndarray = np.ndarray,
    V_ek_int: np.ndarray = np.ndarray,
    alpha: np.ndarray = np.ndarray,
) -> np.ndarray:

    real_u_ek = np.empty_like(U_ek_int)
    real_v_ek = np.empty_like(U_ek_int)

    for i in range(len(alpha)):
        for j in range(5):

            T = rot_matrix(alpha[i])

            vel = np.empty([2, 1])
            vel[0, 0] = U_ek_int[i, j]
            vel[1, 0] = V_ek_int[i, j]

            real_u_ek[i, j] = np.matmul(T, vel)[0]
            real_v_ek[i, j] = np.matmul(T, vel)[1]

    return real_u_ek, real_v_ek


# windstress.py

import numpy as np


def windstress(U: np.ndarray = np.ndarray, Cd: np.ndarray = np.ndarray) -> np.ndarray:

    # input
    # U: absolute wind speed UV_abs
    # Cd: drag coefficient of absolute wind speed

    # Set parameters
    rho_air = np.ones_like(U, dtype="float64") * 1.2041  # kg/m³

    tau = np.ones_like(Cd, dtype="float64")

    # Calculate windstress
    for i in range(5):
        tau[:, i] = rho_air * Cd[:, i] * U ** 2

    return tau


# Define Drag Coefficients extend ranges

# References
#    ----------
#    .. [LP81]
#        | Large and Pond, 1981.
#        | `https://doi.org/10.1175/1520-0485(1981)011<0324:OOMFMI>2.0.CO;2`
#    .. [T90]
#        | Trenberth et al., 1990.
#        | `https://doi.org/10.1175/1520-0485(1990)020<1742:TMACIG>2.0.CO;2`
#    .. [YT96]
#        | Yelland and Taylor, 1996.
#        | `https://doi.org/10.1175/1520-0485(1996)026<0541:WSMFTO>2.0.CO;2`
#    .. [LY04]
#        | Large and Yeager, 2004.
#        | `http://dx.doi.org/10.5065/D6KK98Q6`
#    .. [KH07]
#        | *A note on parameterizations of the drag coefficient*.
#        | A. Köhl and P. Heimbach, August 15, 2007.


import numpy as np

# ncep_ncar_2007
def ncep_ncar_2007_ext(U) -> np.ndarray:

    Cd = np.ones_like(U) * 1.3e-3
    return Cd


# large_and_pond_1981
def large_and_pond_1981_ext(
    U: np.ndarray = np.ndarray, extend_ranges=True
) -> np.ndarray:

    Cd = np.empty(U.shape)

    Cd = 1.2e-3 * (U < 11) + (0.49 + 0.065 * U) * 1e-3 * (U > 11)

    if not extend_ranges:
        Cd = np.where(np.logical_and(4 <= U, U <= 25), Cd, np.nan)
        return Cd
    elif extend_ranges:
        return Cd


# yelland_and_taylor_1996
def yelland_and_taylor_1996_ext(
    U: np.ndarray = np.ndarray, extend_ranges=True
) -> np.ndarray:

    Cd = np.empty(U.shape)

    epsilon = 1.0e-24

    Cd = (0.29 + 3.1 / (U + epsilon) + (7.7 / ((U + epsilon) ** 2))) * (
        U < 6
    ) * 1e-3 + (0.6 + 0.07 * U) * (U >= 6) * 1e-3

    if not extend_ranges:
        Cd = np.where(np.logical_and(3 <= U, U <= 26), Cd, np.nan)
        return Cd
    elif extend_ranges:
        return Cd


# trenberth_etal_1990
def trenberth_etal_1990_ext(U: np.ndarray = np.ndarray) -> np.ndarray:

    Cd = np.empty(U.shape)

    epsilon = 1.0e-24

    Cd = (
        2.18e-3 * (U <= 1)
        + (0.62 + 1.56 / (U + epsilon)) * 1.0e-3 * np.logical_and(1 < U, U <= 3)
        + 1.14e-3 * np.logical_and(3 < U, U < 10)
        + (0.49 + 0.065 * U) * 1.0e-3 * (10 <= U)
    )

    return Cd


# large_and_yeager_2004
def large_and_yeager_2004_ext(
    U: np.ndarray = np.ndarray, extend_ranges=True
) -> np.ndarray:

    Cd = np.empty(U.shape)

    epsilon = 1.0e-24

    Cd = ((0.142 + 0.076 * U + 2.7 / (U + epsilon))) * 1e-3

    if not extend_ranges:
        Cd = np.where((U != 0), Cd, np.nan)
        return Cd
    elif extend_ranges:
        return Cd


# Define Drag Coefficients no extend ranges

# References
#    ----------
#    .. [LP81]
#        | Large and Pond, 1981.
#        | `https://doi.org/10.1175/1520-0485(1981)011<0324:OOMFMI>2.0.CO;2`
#    .. [T90]
#        | Trenberth et al., 1990.
#        | `https://doi.org/10.1175/1520-0485(1990)020<1742:TMACIG>2.0.CO;2`
#    .. [YT96]
#        | Yelland and Taylor, 1996.
#        | `https://doi.org/10.1175/1520-0485(1996)026<0541:WSMFTO>2.0.CO;2`
#    .. [LY04]
#        | Large and Yeager, 2004.
#        | `http://dx.doi.org/10.5065/D6KK98Q6`
#    .. [KH07]
#        | *A note on parameterizations of the drag coefficient*.
#        | A. Köhl and P. Heimbach, August 15, 2007.


import numpy as np

# ncep_ncar_2007
def ncep_ncar_2007(U) -> np.ndarray:

    Cd = np.ones_like(U) * 1.3e-3
    return Cd


# large_and_pond_1981
def large_and_pond_1981(U: np.ndarray = np.ndarray, extend_ranges=False) -> np.ndarray:

    Cd = np.empty(U.shape)

    Cd = 1.2e-3 * (U < 11) + (0.49 + 0.065 * U) * 1e-3 * (U > 11)

    if not extend_ranges:
        Cd = np.where(np.logical_and(4 <= U, U <= 25), Cd, np.nan)
        return Cd
    elif extend_ranges:
        return Cd


# yelland_and_taylor_1996
def yelland_and_taylor_1996(
    U: np.ndarray = np.ndarray, extend_ranges=False
) -> np.ndarray:

    Cd = np.empty(U.shape)

    epsilon = 1.0e-24

    Cd = (0.29 + 3.1 / (U + epsilon) + (7.7 / ((U + epsilon) ** 2))) * (
        U < 6
    ) * 1e-3 + (0.6 + 0.07 * U) * (U >= 6) * 1e-3

    if not extend_ranges:
        Cd = np.where(np.logical_and(3 <= U, U <= 26), Cd, np.nan)
        return Cd
    elif extend_ranges:
        return Cd


# trenberth_etal_1990
def trenberth_etal_1990(U: np.ndarray = np.ndarray) -> np.ndarray:

    Cd = np.empty(U.shape)

    epsilon = 1.0e-24

    Cd = (
        2.18e-3 * (U <= 1)
        + (0.62 + 1.56 / (U + epsilon)) * 1.0e-3 * np.logical_and(1 < U, U <= 3)
        + 1.14e-3 * np.logical_and(3 < U, U < 10)
        + (0.49 + 0.065 * U) * 1.0e-3 * (10 <= U)
    )

    return Cd


# large_and_yeager_2004
def large_and_yeager_2004(
    U: np.ndarray = np.ndarray, extend_ranges=False
) -> np.ndarray:

    Cd = np.empty(U.shape)

    epsilon = 1.0e-24

    Cd = ((0.142 + 0.076 * U + 2.7 / (U + epsilon))) * 1e-3

    if not extend_ranges:
        Cd = np.where((U != 0), Cd, np.nan)
        return Cd
    elif extend_ranges:
        return Cd


# EKMAN_TRANSPORT.PY
import numpy as np


def Ekman_Transport(
    U: np.ndarray = np.ndarray,
    V: np.ndarray = np.ndarray,
    LAT: np.ndarray = np.ndarray,
    drag_coeff=5,
    extend_ranges=False,
    ekman_layer_constant=True,
) -> np.ndarray:

    """
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
    """

    # 1. Check and reshape Data
    U_vec, V_vec, LAT_vec, UV_abs = check_and_reshape(
        U=U, V=V, LAT=LAT
    )  # Reshapes all input data into vectors

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
    ekman_layer_depth_array, f_coriolis_array = reshape_outputs(
        tau=tau, ekman_layer_depth=ekman_layer_depth, f_coriolis=f_coriolis
    )

    # 6. Compute angle to purely east
    alpha = angle_to_pure_east(U=U_vec, V=V_vec)

    # 7. Compute vertically integrated Ekman transport assuming wind is blowing only in positive u direction
    ekman_transp_u, ekman_transp_v = ekman_transport_calc(
        tau=tau,
        ekman_layer_depth_array=ekman_layer_depth_array,
        f_coriolis_array=f_coriolis_array,
        alpha=alpha,
    )

    # 8. Rotate Ekman transport to true direction
    real_u_ek, real_v_ek = rotate_vector(
        U_ek_int=ekman_transp_u, V_ek_int=ekman_transp_v, alpha=alpha
    )  # m²/s

    # 9. Reshape data into original shape
    U_ek_int_reshaped, V_ek_int_reshaped, LAT_reshaped = reshape_results(
        U=U, U_ek_int=real_u_ek, V_ek_int=real_v_ek, LAT_vec=LAT_vec
    )


    # 10. Rename data
    U_ekman = U_ek_int_reshaped
    V_ekman = V_ek_int_reshaped

    # 11. Give Output

    if drag_coeff == 0:
        print("Drag Coefficient: ncep_ncar_2007")
        return (U_ekman[:, :, 0], V_ekman[:, :, 0])

    elif drag_coeff == 1:
        print("Drag Coefficient: large_and_pond_1981")
        return (U_ekman[:, :, 1], V_ekman[:, :, 1])

    elif drag_coeff == 2:
        print("Drag Coefficient: trenberth_etal_1990")
        return (U_ekman[:, :, 0], V_ekman[:, :, 0])

    elif drag_coeff == 3:
        print("Drag Coefficient: yelland_and_taylor_1996")
        return (U_ekman[:, :, 3], V_ekman[:, :, 3])

    elif drag_coeff == 4:
        print("Drag Coefficient: large_and_yeager_2004")
        return (U_ekman[:, :, 4], V_ekman[:, :, 4])

    elif drag_coeff == 5:
        U_ekman, V_ekman = add_mean(U_ekman,V_ekman
        )
        print("Drag Coefficient: [:,:,0] ncep_ncar_2007")
        print("Drag Coefficient: [:,:,1] large_and_pond_1981")
        print("Drag Coefficient: [:,:,2] trenberth_etal_1990")
        print("Drag Coefficient: [:,:,3] yelland_and_taylor_1996")
        print("Drag Coefficient: [:,:,4] large_and_yeager_2004")
        print("Drag Coefficient: [:,:,5] mean")

        return (U_ekman[:, :, :], V_ekman[:, :, :])
