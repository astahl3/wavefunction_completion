import numpy as np
#import matplotlib as mpl
from genEigenstates import genEigenstates
import compHelperFunctions as hf
from genLocalHams import genLocalHams
from applyHam import applyHam
from scipy.sparse.linalg import eigsh, LinearOperator
from wavefunctionCompEx import doCompletion # DEBUGGING ONLY

# Parameters
ham_type = 'rand-homog-r' # type of Hamiltonian (options in genLocalHams.py)
N = 14 # number of lattice sites
d = 2 # local dimension of each lattice site
n = 2 # interaction length of local operators
use_pbc = False # periodic boundary conditions
gam = 1 # for model="Ising" (see genLocalHams.py)
lam = 1 # for model="Ising" (see genLocalHams.py)
Enum = 1 # desired eigenstate (1 = ground state, 2 = first excited, ...)
sample_rate = 0.80
data_loc = '/Users/astahl/repositories/wavefunction_completion/ML_training_data/N14_d2_n2_OBC_s80_t1000_fixedLocs.npz'
num_iters = 5000
num_test_iters = 200

# Generate batch data for training the network
def generate_data(iters, sample_rate):
    
    states = np.zeros((iters,d**N))
    incomplete_states = np.zeros((iters,d**N))
    
    for i in range(0, iters):
        hloc = genLocalHams(ham_type, N, use_pbc, d, n)
        
        # Cast the Hamiltonian 'H' as a linear operator
        def applyLocalHamClosed(psiIn):
            return applyHam(psiIn, hloc, N, use_pbc, d, n)
        H = LinearOperator((d**N, d**N), matvec=applyLocalHamClosed)

        # Perform the exact diagonalization
        E, psi = eigsh(H, k=Enum, which='SA')
        print('Eigenstate number generated: %d with N: %d and energy %.4e'  
                  % (i, N, E[Enum-1]))
        
        if Enum > 1:
            psi = psi[:,Enum-1]
            
        # Initialize incomplete, complete states
        psi_inc, _ = hf.genIncompleteState(psi, sample_rate, slocs) 
        states[i,:] = psi.reshape(d**N,)
        incomplete_states[i,:] = psi_inc.reshape(d**N,)
        
        # DEBUGGING - COMPLETE STATE
        '''
        all_locs = np.arange(psi.size)
        ulocs = np.setdiff1d(all_locs, slocs)
        psi_comp, fid_err = doCompletion(psi, psi_inc, ulocs, N, d, n, use_pbc)
        '''
        
    print(f'Finished generating {iters} eigenstates for testing')
    
    return incomplete_states, states


# Use fixed sample locations
total_entries = d**N
num_samples = int(np.floor(sample_rate * total_entries))
slocs = np.random.choice(range(total_entries), 
                                size=num_samples, replace=False)
psi_mask = np.ones((total_entries,))
psi_mask[slocs] = 0

# Data Generation
train_incomplete, train_complete = generate_data(num_iters, sample_rate)
test_incomplete, test_complete = generate_data(num_test_iters, sample_rate)

# Save data to training folder
np.savez(data_loc, train_incomplete=train_incomplete, train_complete=train_complete,
             test_incomplete=test_incomplete, test_complete=test_complete, psi_mask=psi_mask)
