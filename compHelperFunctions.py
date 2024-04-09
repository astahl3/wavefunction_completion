'''
Several helper functions for use in wavefunction completion


By: Aaron Stahl (2023)

'''
import numpy as np
from typing import Tuple


def normalizeOverUnknowns(psi_in: np.ndarray, ulocs: np.ndarray) -> np.ndarray:
    '''
    Enforce normalization of passed quantum state "psi_in" by scaling only the
    unknown (unsampled) coefficients located at "ulocs"

    PARAMETERS:
    -----------
    psi_in         = wavefunction to be normalized 
    ulocs          = indices of the unsampled (unknown) wavefunction entries
    
    RETURNS:
    --------
    psi_in         = original wavefunction with adjusted magnitude
    
    '''
    # Determine sampled index locations
    locs = np.arange(psi_in.size)
    klocs = np.setdiff1d(locs, ulocs) 
    
    # Calculate norms of known (sampled) and unknown coefficients
    norm_psi_k = np.linalg.norm(psi_in.flat[klocs])
    norm_psi_u = np.linalg.norm(psi_in.flat[ulocs])
    
    # Normalize < psi_in | psi_in > = 1 quantum state over unsampled entries
    psi_in.flat[ulocs] = (np.sqrt(1-norm_psi_k**2)
                          / norm_psi_u)*psi_in.flat[ulocs]
    return psi_in

    
    
def stateFidelity(psi_A: np.ndarray, psi_B: np.ndarray):
    '''
    Calculates fidelity between two quantum states "psi_A" and "psi_B", 
    typically the trial (or completed) wavefunction, and the true wavefunction
    
    PARAMETERS:
    -----------
    psi_A          = wavefunction A
    psi_B          = wavefunction B
    
    RETURNS:
    --------
    fid            = fidelity between states
    
    '''
    size_A = psi_A.shape
    size_B = psi_B.shape
    
    psi_A = np.reshape(psi_A, (np.prod(size_A), 1))
    psi_B = np.reshape(psi_B, (np.prod(size_B), 1))
    
    fid_numer = psi_A.conj().T @ psi_B + psi_B.conj().T @ psi_A
    fid_denom = psi_A.conj().T @ psi_A + psi_B.conj().T @ psi_B
    
    fid = abs(fid_numer / fid_denom)
    
    return fid

def genIncompleteState(psi: np.ndarray, sample_rate: float) \
                            -> Tuple[np.ndarray, np.ndarray]:
    '''
    Creates an incomplete wavefunction by randomly sampling the fraction of 
    coefficients of state "psi" given by "sample_rate" 

    PARAMETERS:
    -----------
    psi            = the original wavefunction
    sample_rate    = the fraction of entries to randomly sample (e.g., 0.20 )

    RETURNS:
    --------
    psi_incomplete = the incomplete wavefunction with sampled entries retained
                     and the rest set to zero.
    slocs          = indices of the sampled entries
    
    '''
    psi = np.array(psi)
    total_entries = psi.size
    num_samples = int(np.floor(sample_rate * total_entries))
    slocs = np.random.choice(range(total_entries), 
                                    size=num_samples, replace=False)
    
    psi_incomplete = np.zeros_like(psi)
    psi_incomplete[slocs] = psi[slocs]
    
    return psi_incomplete, slocs
