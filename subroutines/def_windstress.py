import numpy as np

def windstress(U,V,Cd_U,Cd_V):
# Set parameters
rho_air = np.ones_like(U) * 1.2041 #kg/m³

taux = np.empty_like(Cd_U)
tauy = np.empty_like(Cd_U)

# Calculate windstress
for i in range(5):
    taux[:,i] = rho_air * Cd_U[:,i] * U**2
    tauy[:,i] = rho_air * Cd_V[:,i] * V**2

    return(taux, tauy)
