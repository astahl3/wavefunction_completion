'''
Includes several functions required for tensor tree completions with randomly
structured trees
'''

import numpy as np

def genBlocksTree(N, sz):
    if N == 1:
        Bs = np.array([1])
        dims_new = np.array([sz])
        
    else:
        Bs = np.array([1])
        while np.max(Bs) < 2: # ensure array isn't all 1s (i.e., trivial shape)
            
            # Generates array with 50/50 split between 1 and 2
            Bs = np.floor(2 * np.random.rand(N,1)) + 1
            #Bs = np.random.randint(1,3,[N,1])
            
            cs_Bs = np.cumsum(Bs, axis=0)
            Bs = Bs[0:np.sum(cs_Bs <= N)]
            if sum(Bs) < N:
                Bs = np.append(Bs, 1)
                
        upd_cs = np.append(0, np.cumsum(Bs))
        dims_new = np.array([], dtype=int)
        for k in range(0, len(upd_cs)-1):
            app_shape = np.prod(sz[int(upd_cs[k]):int(upd_cs[k+1])])
            dims_new = np.append(dims_new, app_shape)
            
        return dims_new


def oneLayerTree(psi_in, shape, chi_max):
    wiso = []
    wlist = []
    shape_out = shape.copy()
    
    for k in range(0, len(psi_in.shape)):
        if shape[k] > chi_max:
            wlist.append(k)
            d1 = int(np.prod(shape_out[:k]))
            d2 = shape_out[k]
            d3 = int(np.prod(shape_out[k+1:]))
            
            # Reshape psi as [d1, d2, d3], then permute target dim to front
            psi_in = np.reshape(psi_in, (d1, d2, d3))
            psi_in = np.transpose(psi_in, (1,0,2))
            psi_in = np.reshape(psi_in, (d2, d1*d3))
            
            # Perform truncated SVD, save U and SV'
            ut, st, vt = np.linalg.svd(psi_in, full_matrices=False)
            
            chi_temp = min(chi_max, st.shape[0])
            wiso.append(ut[:,0:chi_temp])
            shape_out[k] = chi_temp
            
            st_diag = np.diag(st[:chi_temp])
            st_vt = st_diag @ vt[:chi_temp, :]
            st_vt = np.reshape(st_vt, (chi_temp, d1, d3))
            st_vt = np.transpose(st_vt, (1, 0, 2))
            psi_in = np.reshape(st_vt, shape_out)
            
    return psi_in, wlist, wiso, shape_out


def reverseLayerTree(psi_fin, shape_mid, wlist, wiso):
    for p in range(0, len(wlist)):
        k = wlist[p]
        d1 = int(np.prod(shape_mid[:k]))
        d2 = shape_mid[k]
        d3 = int(np.prod(shape_mid[k+1:]))
        
        # Reshape and permute psi_fin tensor
        psi_fin = np.reshape(psi_fin, (d1, d2, d3))
        psi_fin = np.transpose(psi_fin, (1, 0, 2))
        psi_fin = np.reshape(psi_fin, (d2, d1*d3))
        
        # Multiply with wiso tensor
        psi_temp = wiso[p] @ psi_fin
        chi0 = wiso[p].shape[0]
        shape_mid[k] = chi0
        
        # Reshape and permute back
        psi_temp = np.reshape(psi_temp, (chi0, d1, d3))
        psi_temp = np.transpose(psi_temp, (1, 0, 2))
        psi_temp = np.reshape(psi_temp, shape_mid)
        
        psi_fin = psi_temp
        
    return psi_fin