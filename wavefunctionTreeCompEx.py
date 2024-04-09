import numpy as np
from typing import Tuple
from scipy.sparse.linalg import LinearOperator, eigsh
from timeit import default_timer as timer
from truncatedMPS import truncatedMPS
from applyHam import applyHam
from genLocalHams import genLocalHams
from stateFidelity import stateFidelity
import treeCompFunctions as tc
import compHelperFunctions as hf

def main():
    
    # Simulation parameters
    model = 'rand-homog-c' # type of Hamiltonian (options in genLocalHams.py)
    N = 16 # number of lattice sites
    d = 2 # local dimension of each lattice site
    n = 2 # interaction length of local operators
    use_pbc = False # periodic boundary conditions
    gam = 1 # for model="Ising" (see genLocalHams.py)
    lam = 1 # for model="Ising" (see genLocalHams.py)
    Enum = 1 # desired eigenstate (1 = ground state, 2 = first excited, ...)

    # Completion parameters
    chi_min = 2 # minimum bond dimension
    chi_max = 60 # maximum bond dimension
    chi_step = 1 # step between bond dimensions
    max_iter = 200 # iterations per bond dimension
    sample_rate = 0.2 # fraction of quantum state to be sampled (randomly)
    err_tol = 1e-8 # required fidelity error for completed state
    
    # Generate desired eigenstate and initialize "incomplete" state
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

    psi_inc, slocs = hf.genIncompleteState(psi, sample_rate)
    all_locs = np.arange(psi.size)
    ulocs = np.setdiff1d(all_locs, slocs)
    
    # Perform tensor tree completion
    fid_err = []
    psi_trial = np.copy(psi_inc)
    psi_trial = psi_trial.reshape((d,)*N) # reshape as N-dim tensor 
    
    # Iterater over bond dimension "chi"
    for chi in range(chi_min, chi_max+1, chi_step):
        for k in range(0, max_iter):
            err_temp = 1 - stateFidelity(psi_trial, psi)
            fid_err.append(err_temp)

            pivot = np.random.randint(N)
            perm_vec = np.concatenate((np.arange(pivot, N), 
                                       np.arange(0,pivot)))
            
            wlistK= [None] * N
            wisoK = [None] * N
            dimsK = [None] * N
            psi_temp = np.transpose(psi_trial, perm_vec)
            
            for x in range(0, N+1):
                Nsize = psi_temp.shape
                Ntemp = len(Nsize)
                
                # Generate random block groupings
                dims = tc.genBlocksTree(Ntemp, Nsize)
                
                # Build tree network
                psi_temp = np.reshape(psi_temp, dims)
                psi_temp, wlistK[x], wisoK[x], dimsK[x] = \
                        tc.oneLayerTree(psi_temp, dims, chi) 
                
                if psi_temp.ndim < 3:
                    xmax = x
                    break
            
            # Rebuild original state from truncated tensor tree network
            for x in range(xmax, -1, -1):
                psi_temp = tc.reverseLayerTree(psi_temp, dimsK[x], 
                                               wlistK[x], wisoK[x])
            
            # Update unknown state coefficients in psi_trial
            psi_temp = np.reshape(psi_temp, (d,) * N)
            psi_temp = np.transpose(psi_temp, np.argsort(perm_vec))
            psi_trial.flat[ulocs] = psi_temp.flat[ulocs]
            
            # Normalize over unknown coefficients
            psi_trial = hf.normalizeOverUnknowns(psi_trial, ulocs)
            psi_trial = psi_trial.reshape((d,)*N)

        fid_err_temp = fid_err[-1]
        print('bond dim: %2.0d, fidelity error: %.2e' % (chi, fid_err_temp))
        if fid_err_temp < err_tol:
            break

    print('*********************************')
    print('Wavefunction completion finished:')
    print('*********************************')
    print('fidelity error = %.2e at bond dimension chi = %2.0d' 
          % (fid_err_temp, chi))

if __name__ == "__main__":
    main()