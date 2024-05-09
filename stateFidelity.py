import numpy as np

def stateFidelity(psi_A, psi_B):
    size_A = psi_A.shape
    size_B = psi_B.shape
    
    psi_A = np.reshape(psi_A, (np.prod(size_A), 1))
    psi_B = np.reshape(psi_B, (np.prod(size_B), 1))
    
    fid_numer = psi_A.conj().T @ psi_B + psi_B.conj().T @ psi_A
    fid_denom = psi_A.conj().T @ psi_A + psi_B.conj().T @ psi_B
    
    fid = abs(fid_numer / fid_denom)
    
    return fid