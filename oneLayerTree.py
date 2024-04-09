import numpy as np

def oneLayerTree(psi_in, shape, chi_max):
    wiso = []
    wlist = []
    dims_out = shape.copy()
    
    for k in range(0, len(psi_in.shape)):
        if shape[k] > chi_max:
            wlist.append(k)
            d1 = int(np.prod(dims_out[:k]))
            d2 = dims_out[k]
            d3 = int(np.prod(dims_out[k+1:]))
            
            # Reshape psi as [d1, d2, d3], then permute target dim to front
            psi_in = np.reshape(psi_in, (d1, d2, d3))
            psi_in = np.transpose(psi_in, (1,0,2))
            psi_in = np.reshape(psi_in, (d2, d1*d3))
            
            # Perform truncated SVD, save U and aborb SV'
            ut, st, vt = np.linalg.svd(psi_in, full_matrices=False)
            chi_temp = min(chi_max, st.shape[0])
            wiso.append(ut[:,0:chi_temp])
            dims_out[k] = chi_temp
            
            st_diag = np.diag(st[:chi_temp])
            st_vt = st_diag @ vt[:chi_temp, :]
            st_vt = np.reshape(st_vt, (chi_temp, d1, d3))
            st_vt = np.transpose(st_vt, (1, 0, 2))
            psi_in = np.reshape(st_vt, dims_out)
            
    return psi_in, wlist, wiso, dims_out