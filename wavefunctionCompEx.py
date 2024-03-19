import numpy as np
from typing import Tuple
from scipy.sparse.linalg import LinearOperator, eigsh
from timeit import default_timer as timer
from truncatedMPS import truncatedMPS
from applyHam import applyHam
from genLocalHams import genLocalHams

def genIncompleteState(psi: np.ndarray, sample_rate: float) \
                            -> Tuple[np.ndarray, np.ndarray]:
    
    '''
    Creates an incomplete version of the wavefunction by randomly sampling 
    entries.

    PARAMETERS:
    psi            = the original wavefunction
    sample_rate    = the fraction of entries to sample (e.g., 0.20 for 20%)

    RETURNS:
    psi_incomplete = the incomplete wavefunction with sampled entries retained
                     and the rest set to zero.
    sampled_locs   = indices of the sampled entries
    
    '''
    psi = np.array(psi)
    total_entries = psi.size
    num_samples = int(np.floor(sample_rate * total_entries))
    sampled_locs = np.random.choice(range(total_entries), 
                                    size=num_samples, replace=False)
    
    psi_incomplete = np.zeros_like(psi)
    psi_incomplete[sampled_locs] = psi[sampled_locs]
    
    return psi_incomplete, sampled_locs

def main():
    
    # Simulation parameters
    model = 'rand-homog-c'
    Nsites = 16
    d = 2
    n = 2
    usePBC = False
    gam = 1
    lam = 1
    Enum = 1

    # Completion parameters
    rec_method = 'MPS'
    chi_min = 2
    chi_max = 60
    chi_step = 1
    max_iter = 60
    sample_rate = 0.2
    norm_lvl = 'iter'
    
    # Generate desired eigenstate and initialize "incomplete" state
    hloc = genLocalHams(model, Nsites, usePBC, d, n)
    
    # Cast the Hamiltonian 'H' as a linear operator
    # This defines the matrix (Hamiltonian) by its action on a vector rather than 
    # by its explicit matrix entries. This makes solving for eigenvalues much
    # more efficient and practical for large matrices
    def applyLocalHamClosed(psiIn):
        return applyHam(psiIn, hloc, Nsites, usePBC, d, n)
    
    H = LinearOperator((d**Nsites, d**Nsites), matvec=applyLocalHamClosed)

    # Perform the exact diagonalization
    start_time = timer()
    E, psi = eigsh(H, k=Enum, which='SA')
    diag_time = timer() - start_time
    print('Eigenstate generated with N: %d, time: %.2f s, energy %.4e'  
              % (Nsites, diag_time, E[Enum-1]))

    
    if Enum > 1:
        psi = psi[:,Enum]
    psi_inc, sampled_locs = genIncompleteState(psi, sample_rate)
    all_locs = np.arange(psi.size)
    unknown_locs = np.setdiff1d(all_locs, sampled_locs)
    
    # Perform the completion
    psi_rec = np.copy(psi_inc)
    for j in range(chi_min, chi_max+1, chi_step):
        for k in range(0, max_iter):
            psi_rec = truncatedMPS(psi_rec, unknown_locs, j, Nsites, d)
            fid_err = 1 - np.abs((np.dot(psi_rec.conj().T, psi)*np.dot(psi.conj().T, psi_rec)) \
                          / (np.dot(psi_rec.conj().T, psi_rec)*np.dot(psi.conj().T, psi)))
        
        print('Bond dim: %2.0d, fidelity error: %.2e' % (j, fid_err))

    print('Wavefunction completion finished, fidelity error = %.2e at \
              bond dimension chi = %2.0d' % (fid_err, j))
if __name__ == "__main__":
    main()