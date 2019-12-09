# Ekman Transport Users Manual

This repository contains a set of functions to calculate the zonal and meridional Ekman Transport [m²/2] from a given wind field using different drag coefficient bulk formulas to compute wind stress from absolute wind speed. The Ekman transport is calculated by integrating Ekman spiral velocities from the Ekman depth to the surface. The depth of the Ekman layer (water column under wind influence) can be either set to constant (100m) or varying in space and time dependent on local wind speed and latitude.
Furthermore the drag coefficients can be forced to extrapolate into ranges not covered by the bulk formulas.

## Usage:
  1. Download the repository to your local machine
  2. import the package ```from ekman_transport.functions import *```
  3. run ``` `U_ek, V_ek = Ekman_Transport(...)´ ```
  
### Input:
1. ```U:     [n x m] zonal wind speed ```
2. ```V:     [n x m] merdidional wind speed ```
3. ```LAT:   [n x m] latitude ```

### Options:
  1. `drag_coeff = 0 | 1 | 2 | 3 | 4 | 5 (default)`  
  1.1 [0] ncep_ncar_2007  
  1.2 [1] large_and_pond_1981  
  1.3 [2] trenberth_etal_1990  
  1.4 [3] yelland_and_taylor_1996  
  1.5 [4] large_and_yeager_2004  
  1.6 [5] all  
  2. `ekman_layer_constant = True | False `  
  2.1 True: 100m, constant in space and time  
  2.2 False: Vaying in space and time with total wind speed  
  3. `extend_ranges = True | False`  
  3.1 True: In case drag coefficient does not cover certain wind speeds, extrapolate drag coefficient  
  3.2 False: In case drag coefficient does not cover certain wind speeds, mask with `np.nan`  
  
### Copyright
This package is provided as it is, but it's mine so you can't sell it.   
(c) Finn Heukamp


