import numpy as np

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