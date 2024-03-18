import numpy as np

def applyHam(psi: np.ndarray, hloc: list[np.ndarray], N: int, 
                   usePBC: bool, d: int, n: int) -> np.ndarray:
    '''
    Applies local Hamiltonian operators (given as a set of local terms, 'hloc') 
    to input quantum state 'psi'

    PARAMETERS:             
    psi      = input quantum state
    hloc     = 1D array of local Hamiltonians; cell{1} corresponds to a local 
               Hamiltonian operator that couples the first lattice site to the
               next (n-1) sites
    N        = total nubmer of lattice sites
    usePBC   = set to True for periodic boundary conditions, False for open
               boundary conditions
    d        = local dimension for each lattice site (e.g., d = 2 for a qubit)
    n        = number of sites spanned by local Hamiltonian terms (i.e., the 
               interaction or coupling length)
    
    RETURNS:
    psi_sum  = state vector after applying all local Hamiltonian terms
    
    '''  
    if np.iscomplexobj(hloc[0]):
        psi_sum = np.zeros(psi.size, dtype=np.complex128)
    else:
        psi_sum = np.zeros(psi.size, dtype=psi.dtype)
    
    # Apply local Hamiltonian terms to non-boundary lattice sites
    for i in range(0, (N - (n - 1))):
        h = hloc[i].reshape(d**n, d**n)
        d1 = i
        d2 = n
        d3 = N - d1 - d2
        
        
        # VER. A
        # Apply local operator "in place" after partitioning around the lattice 
        # sites to be operated on
        
        psi_temp = psi.reshape(d**d1, d**d2, d**d3)
        psi_temp = np.tensordot(h, psi_temp, axes=[[1],[1]])
        psi_temp = psi_temp.transpose(1,0,2)
        psi_temp = psi_temp.reshape(d**N,)
        psi_sum += psi_temp
        '''
        # VER. B
        # Permute indices such that sites to be operated on are swapped with  
        # the first dimension (visually, bring the lattice sites to be operated 
        # on to the left-most side of the wavefunction)
        
        psi_temp = psi.reshape(d**d1, d**d2, d**d3)
        psi_temp = psi_temp.transpose(1,0,2)
        psi_temp = np.tensordot(h,psi_temp,axes=[[1],[0]])
        psi_temp = psi_temp.transpose(1,0,2)
        psi_temp = psi_temp.reshape(d**N,)
        psi_sum += psi_temp
        '''
        '''
        # VER. C
        # Same as ver. B, but perform the reshaping manually so as to permit
        # the direct call of numpy.dot()
        
        psi_temp = psi.reshape(d**d1, d**d2, d**d3)
        psi_temp = psi_temp.transpose(1,0,2)
        psi_temp = psi_temp.reshape(d**n, d**(N-n))
        psi_temp = np.dot(h, psi_temp)
        psi_temp = psi_temp.reshape(d**d2, d**d1, d**d3)
        psi_temp = psi_temp.transpose(1,0,2)
        psi_temp = psi_temp.reshape(d**N,)
        psi_sum += psi_temp
        '''
        
    # If using periodic boundary conditions, apply local Hamiltonian terms to
    # the boundary lattice sites
    if usePBC:
        
        
        # VER. A
        # Analogous to ver. A above; apply local operator "in place" after
        # partitioning sites to be operated on (i.e., boundary sites)
        k = 1
        for j in range(N - (n - 1), N):
            h = hloc[i].reshape(d, d, d, d)
            psi_temp = psi.reshape(d**k, d**(N - n), d**(n - k))
            psi_temp = np.tensordot(h, psi_temp, axes=[[2,3],[2,0]])
            psi_temp = psi_temp.transpose(1,2,0)
            psi_temp = psi_temp.reshape(d**N,)
            psi_sum += psi_temp
            
        '''
        # VER. C
        # Analogous to ver. C above; rotate last k sites to front of lattice, 
        # then reshape to apply the local operator
        k = 1
        for j in range(N - (n - 1), N):
            h = hloc[i].reshape(d**n, d**n)
            psi_temp = psi.reshape(d**k, d**(N - n), d**(n - k))
            psi_temp = psi_temp.transpose(2,0,1)
            psi_temp = psi_temp.reshape(d**n, d**(N - n))
            psi_temp = np.dot(h, psi_temp)
            psi_temp = psi_temp.reshape(d**(n - k), d**k, d**(N - n))
            psi_temp = psi_temp.transpose(1,2,0)
            psi_temp = psi_temp.reshape(d**N,)
            psi_sum += psi_temp
            k = k + 1
        '''
    return psi_sum