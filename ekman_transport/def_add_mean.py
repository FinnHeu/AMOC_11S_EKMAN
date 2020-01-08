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
