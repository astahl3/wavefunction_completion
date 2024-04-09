import numpy as np
import warnings
from scipy.linalg import svd
from ncon import ncon

def truncatedMPS(psi_in: np.ndarray, error_locs: np.ndarray, chi_max: int, 
                 Nsites: int, d: int) -> np.ndarray:
    '''
    For use in wavefunction completion (see wavefnCompletionEx.py)
    
    Generates a matrix product state (MPS) of bond dimension "chi_max" using
    the wavefunction passed as "psi_in", then returns a wavefunction where all 
    locations in "error_locs" are updated according to the MPS, and all other
    entries of the original "psi_in" are unchanged
    
    PARAMETERS:
    -----------
    psi_in         = wavefunction to be approximated via MPS 
    error_locs     = locations of unsampled (or unknown) wavefunction entires
    chi_max        = max bond dimension
    Nsites         = number of lattice sites in model
    d              = local dimension of model
    
    RETURNS:
    --------
    psi_in         = original wavefunction with updated entries
    
    '''
    psi_temp = psi_in

    # Generate reshape vector for SVD cuts and unitaries
    Ntensors = Nsites - 2
    up = [d**k for k in range(1, Ntensors + 1)]
    down = [d*d**k for k in range(Ntensors, 0, -1)]

    dim_vec = [min(min(u, chi_max), min(d, chi_max)) for u, d in zip(up, down)]

    # Generate MPS tensors
    U = [None] * Ntensors
    for k in range(1, Ntensors):
        cut_dim = min(d * dim_vec[k - 1], d * chi_max)
        psi_temp = np.reshape(psi_temp, (cut_dim, d**(Nsites - (k + 1))))
        try:
            u, s, v = svd(psi_temp, full_matrices=False)
        except np.linalg.LinAlgError:
            warnings.warn('SVD failed to converge')
            u, s, v = svd(psi_temp + 1E-12, full_matrices=False)
        
        chi = min(chi_max, s.shape[0])
        U[k - 1] = u[:, :chi]
        psi_temp = np.dot(np.diag(s[:chi]), v[:chi, :])

    U[k] = psi_temp

    # Reshape MPS tensors for ncon call
    for k in range(Ntensors):
        U[k] = np.reshape(U[k], (dim_vec[k], d, dim_vec[Ntensors - k - 1]))

    # Construct ncon indices
    index_array = [[-1, -2, 1]]  # first tensor
    for k in range(2, Ntensors):
        index_array.append([k - 1, -(k + 1), k])
    index_array.append([Ntensors - 1, -(Ntensors + 1), -(Ntensors + 2)])  # last tensor

    # Rebuild eigenstate via ncon
    psi_mps = ncon(U, index_array)
    psi_mps = np.reshape(psi_mps, (d**Nsites, 1))
    psi_in.flat[error_locs] = psi_mps.flat[error_locs]

    return psi_in