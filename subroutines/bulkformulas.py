# Define Drag Coefficients

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
def ncep_ncar_2007(U, LAT) -> np.ndarray:

Cd = np.empty_like(U)
Cd.fill(1.3e-3)

    return Cd


# large_and_pond_1981
def large_and_pond_1981(U: np.ndarray = np.ndarray, extend_ranges=False
) -> np.ndarray:

    Cd = np.empty(U.shape)

    Cd = 1.2e-3 * (U < 11) + (0.49 + 0.065 * U) * 1e-3 * (U > 11)

    if not extend_ranges:
        Cd = np.where(np.logical_and(4 <= U, U <= 25), Cd, np.nan)

        return Cd


# yelland_and_taylor_1996
def yelland_and_taylor_1996(U: np.ndarray = np.ndarray, extend_ranges=False) -> np.ndarray:

    Cd = np.empty(U.shape)

    epsilon = 1.0e-24

    Cd = (0.29 + 3.1 / (U + epsilon) + (7.7 / ((U + epsilon) ** 2))) * (
        U < 6
    ) * 1e-3 + (0.6 + 0.07 * U) * (U >= 6) * 1e-3

    if not extend_ranges:
        Cd = np.where(np.logical_and(3 <= U, U <= 26), Cd, np.nan)

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
