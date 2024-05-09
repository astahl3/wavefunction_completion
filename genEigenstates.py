from scipy.sparse.linalg import eigsh, LinearOperator
from applyHam import applyHam
from genLocalHams import genLocalHams

# Placeholder for doBuildLocalHams and doApplyHam functions
# These functions need to be defined or translated as well
def genEigenstates(iters, sizes, state_nums, ham_type, use_pbc, d, n, **kwargs):
    gam = kwargs.get('gam', 1)
    lam = kwargs.get('lam', 1)    
    # Ensure sizes and state_nums are iterables (lists or tuples)
    if not isinstance(sizes, (list, tuple)):
        sizes = [sizes]
    if not isinstance(state_nums, (list, tuple)):
        state_nums = [state_nums]
    
    states = {}
    hams = {}
    for i in range(iters):
        for size in sizes:
            N = size
            hloc = genLocalHams(ham_type, N, use_pbc, d, n, gam, lam)
            hams[(i, size)] = hloc
            
            # Cast Hamiltonian as a linear operator that operates on the state
            def applyHamClosed(psi):
                return applyHam(psi, hloc, N, use_pbc, d, n)
            H = LinearOperator((d**N, d**N), matvec=applyHamClosed)
            
            for state_num in state_nums:
                E, psi = eigsh(H, k=state_num, which='SA')
                states[(i, size, state_num)] = psi[:, state_num-1]
                print(f'Iteration: {i+1}, Size: {size}, State Number: {state_num}')
    
    return states, hams