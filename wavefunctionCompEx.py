'''
Sample implementation of wavefunction completion on a 1-D lattice

Author: Aaron Stahl (2024)

'''
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from timeit import default_timer as timer
from truncatedMPS import truncatedMPS
from applyHam import applyHam
from genLocalHams import genLocalHams
from allCutSweep import allCutSweep
import compHelperFunctions as hf

def main():
    
    # Simulation parameters
    model = 'rand-homog-c' # type of Hamiltonian (options in genLocalHams.py)
    N = 16 # number of lattice sites
    d = 2 # local dimension of each lattice site
    n = 2 # interaction length of local operators
    use_pbc = True # periodic boundary conditions
    gam = 1 # for model="Ising" (see genLocalHams.py)
    lam = 1 # for model="Ising" (see genLocalHams.py)
    Enum = 1 # desired eigenstate (1 = ground state, 2 = first excited, ...)

    # Recovery method parameter
    '''
    rec_method selects desired completion routine:
        MPS = matrix product state (open boundary conditions)
        ACS = all cut sweep (open and periodic boundary conditions)
    '''
    rec_method = 'ACS'
    
    # Completion parameters
    chi_min = 2 # minimum bond dimension
    chi_max = 60 # maximum bond dimension
    chi_step = 1 # step between bond dimensions
    max_iter = 10 # iterations per bond dimension
    sample_rate = 0.2 # fraction of quantum state to be sampled (randomly)
    err_tol = 1e-8 # required fidelity error for completed state
    
    # Generate desired eigenstate
    hloc = genLocalHams(model, N, use_pbc, d, n)
    
    # Cast the Hamiltonian 'H' as a linear operator
    # This defines the matrix (Hamiltonian) by its action on a vector rather
    # than by its explicit matrix entries. This can make solving for
    # eigenvalues more computationally efficient.
    def applyLocalHamClosed(psiIn):
        return applyHam(psiIn, hloc, N, use_pbc, d, n)
    
    H = LinearOperator((d**N, d**N), matvec=applyLocalHamClosed)

    # Perform the exact diagonalization
    start_time = timer()
    E, psi = eigsh(H, k=Enum, which='SA')
    diag_time = timer() - start_time
    print('Eigenstate generated with N: %d, time: %.2f s, energy %.4e'  
              % (N, diag_time, E[Enum-1]))
    
    if Enum > 1:
        psi = psi[:,Enum]
        
    # Initialize incomplete state
    psi_inc, slocs = hf.genIncompleteState(psi, sample_rate)
    all_locs = np.arange(psi.size)
    ulocs = np.setdiff1d(all_locs, slocs)
    
    # Perform the completion
    psi_trial = np.copy(psi_inc)
    psi_trial = psi_trial.reshape((d,)*N) # reshape as N-dim tensor 

    # Iterater over bond dimension "chi"
    for chi in range(chi_min, chi_max+1, chi_step):
        for k in range(0, max_iter):
            if rec_method == 'MPS':
                psi_trial = truncatedMPS(psi_trial, ulocs, chi, N, d)
            if rec_method == 'ACS':
                psi_trial = allCutSweep(psi_trial, ulocs, chi, N, d, use_pbc)
            
            # Normalize over unknown coefficients
            psi_trial = hf.normalizeOverUnknowns(psi_trial, ulocs)
            psi_trial = psi_trial.reshape((d,)*N)
            
            fid_err = 1 - hf.stateFidelity(psi_trial, psi)
            
        print('bond dim: %2.0d, fidelity error: %.2e' % (chi, fid_err))
        if fid_err < err_tol:
            break
    
    print('*********************************')
    print('Wavefunction completion finished:')
    print('*********************************')
    print('fidelity error = %.2e at bond dimension chi = %2.0d' 
          % (fid_err, chi))

if __name__ == "__main__":
    main()