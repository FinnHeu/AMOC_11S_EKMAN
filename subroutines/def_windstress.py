import numpy as np

def windstress(Cd,U,V)
# Set parameters
rho_air = np.empty_like(Cd)
rho_air.fill(1.2041) #kg/m³

taux = np.empty_like(Cd)
tauy = np.empty_like(Cd)

U_reshaped = np.empty_like(Cd)
V_reshaped = np.empty_like(Cd)

U_reshaped
# Calculate windstress
taux[:,:,i] = rho_air * Cd * U_reshaped**2
tauy[:,:,i] = rho_air * Cd * V_reshaped**2

    return(taux, tauy)
