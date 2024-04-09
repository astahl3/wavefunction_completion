import numpy as np

def allCutSweep(psi_in: np.ndarray, unknown_locs: np.ndarray, chi_max: int, 
                Nsites: int, d: int, use_pbc: bool) -> np.ndarray:
    '''
    Given a trial wavefunction "psi_in", this function performs truncated SVDs
    across all possible nontrivial blocks of size 2:floor(Nsites/2). The 
    unsampled wavefunction coefficients "unknown_locs" are updated after each 
    t-SVD. Treatment for periodic boundary conditions are permitted via 
    "use_pbc". After iterating over all blocks, the updated trial wavefunction
    is returned.
    
    PARAMETERS:
    psi_in         = wavefunction to be approximated via completion method 
    unknown_locs   = locations of unsampled (or unknown) wavefunction entires
    chi_max        = max bond dimension
    Nsites         = number of lattice sites in model
    d              = local dimension of model
    use_pbc        = periodic boundary conditions
    
    RETURNS:
    psi_in         = original wavefunction with updated entries
    
    '''
    # Don't evaluate blocks for which chi_max exceeds the block's degrees of 
    # freedom; e.g., if d = 2 and chi_max = 24, then blocks consisting of 4 or
    # fewer sites (2^4 = 16) won't be truncated and are thus excluded 
    '''
    max_pow = 9
    for k in range(2, max_pow):
        if chi_max >= d**k:
            min_blk_sz = k
            cutoff = False
            break
    '''
    cutoff = False
    k = 2    
    while chi_max >= d**k:
        min_blk_sz = k
        cutoff = True    
        k += 1
        
    if not cutoff:
        min_blk_sz = 2
    
    # Sweep over all nontrivial blocks that permit truncation at the passed
    # bond dimension chi_max
    for blk_sz in range(min_blk_sz, Nsites // 2 + 1):
        for k in range(1, Nsites - blk_sz + 2):
            psi_temp = psi_in.copy()
            
            # Initialize partition indices
            d1 = d**max(0, k-1)
            d2 = d**blk_sz
            d3 = d**max(0, Nsites - blk_sz - (k-1))
            
            # Prepare state for t-SVD
            psi_temp = psi_temp.reshape(d1, d2, d3)
            psi_temp = psi_temp.transpose(1, 0, 2)
            psi_temp = psi_temp.reshape(d2, d1*d3)

            # Perform t-SVD, invert permutation, and update unsampled entries
            ut, st, vt = np.linalg.svd(psi_temp, full_matrices=False)
            chi = min(chi_max, st.shape[0])
            st_diag = np.diag(st[:chi])
            psi_temp = ut[:, :chi] @ st_diag @ vt[:chi, :]
            psi_temp = psi_temp.reshape(d2, d1, d3)
            psi_temp = psi_temp.transpose(1,0,2)
            psi_in.flat[unknown_locs] = psi_temp.flat[unknown_locs]
            
        if use_pbc:
            for k in range(1, blk_sz):
                psi_temp = psi_in.copy()
                
                # Initialize partition indices
                d1 = d**(Nsites - blk_sz + k)
                d2 = d**(blk_sz - k)
                
                # Prepare state for t-SVD
                psi_temp = psi_temp.reshape(d1, d2)
                psi_temp = psi_temp.transpose(1,0)
                psi_temp = psi_temp.reshape(d**blk_sz, -1)

                # Perform t-SVD, invert permutation, update unsampled entries
                ut, st, vt = np.linalg.svd(psi_temp, full_matrices=False)
                chi = min(chi_max, st.shape[0])
                st_diag = np.diag(st[:chi])
                psi_temp = ut[:, :chi] @ st_diag @ vt[:chi, :]
                psi_temp = psi_temp.reshape(d2, d1)
                psi_temp = psi_temp.transpose(1,0)
                psi_in.flat[unknown_locs] = psi_temp.flat[unknown_locs]
    
    return psi_in
