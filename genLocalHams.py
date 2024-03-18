import numpy as np

def genLocalHams(ham_type: str, N: int, usePBC: bool=True, d: int=2, 
                     n: int=2, lam: int=1, gam: int=1) -> list[np.ndarray]:
    """
    Generates a one-dimensional list of local Hamiltonian terms according to 
    desired parameters. For reference on Ising and XY model Hamiltonians, for
    reference, see [Equation 3.1] & [Figure 4] in this work:
        
        J. I. Latorre, E. Rico, and G. Vidal, "Ground state entanglement
        quantum spin chains", Quantum Inf. Comput., 4, pp. 48-92 (2004).
        
        arXiv link: https://arxiv.org/pdf/quant-ph/0304098.pdf

    PARAMETERS:
    ham_type = specifies which Hamiltonian is desired (see options below)
    N        = total number of lattice sites
    usePBC   = set to True for periodic boundary conditions, False for open
               boundary conditions
    d        = local dimension for each lattice site (e.g., d = 2 for a qubit)
    n        = number of sites spanned by local Hamiltonian terms (i.e., the 
               interaction or coupling length)
    lam      = magnetic field strength parameter lambda; only used if the "XY" 
               model is selected
    gam      = spin coupling parameter gamma; only used if the "XY" model is 
               selected
    RETURNS:
    hloc     = list of the local Hamiltonian terms, where the kth entry in hloc
               corresponds to the operator for sites k to (k+n) 


    Acceptable strings for the variable ham_type:    
    
    rand-inhomog-r: random, real, translationally variant local terms
    rand-inhomog-c: random, complex, translationally variant local terms
    rand-homog-r:   random, real, translationally invariant local terms
    rand-homog-c:   random, real, translationally invariant local terms
    XY:             XY spin chain model in 1-D (see reference above)
           
        
    *************************************************
    *       COMMON XY MODEL CONFIGURATIONS          *
    *                                               *
    *   critical XY:      gam >= 0,   lam = 1       *
    *   XX:               gam = 0,    lam >= 0      *
    *   critical XX:      gam = 0,    0 <= lam <= 1 *
    *   Ising:            gam = 1,    lam >= 0      *
    *   critical Ising:   gam = 1,    lam = 1       *
    *************************************************
    
    """
    
    # Set boundary conditions
    if usePBC == 1: num_hams = N  
    else: num_hams = (N - n + 1)
    
    # Initialize the list of local Hamiltonian terms
    hloc = [None] * num_hams
    
    # Initialize Pauli matrices and the identity matrix
    sX = np.array([[0, 1], [1, 0]])
    sY = np.array([[0, -1j], [1j, 0]])
    sZ = np.array([[1, 0], [0, -1]])
    sI = np.eye(2)
    
    # Create local Hamiltonian terms
    if ham_type == 'rand-inhomog-r':
        for k in range(num_hams):
            A = np.random.randn(d**n, d**n)
            hloc[k] = (A + A.T).reshape((d,)*2*n)
            
    elif ham_type == 'rand-inhomog-c':
        for k in range(num_hams):
            A = np.random.randn(d**n, d**n) + 1j * np.random.randn(d**n, d**n)
            hloc[k] = (A + A.conj().T).reshape((d,)*2*n)
            
    elif ham_type == 'rand-homog-r':
        A = np.random.randn(d**n, d**n)
        for k in range(num_hams):
            hloc[k] = (A + A.T).reshape((d,)*2*n)
            
    elif ham_type == 'rand-homog-c':
        A = np.random.randn(d**n, d**n) + 1j * np.random.randn(d**n, d**n)
        for k in range(num_hams):
            hloc[k] = (A + A.conj().T).reshape((d,)*2*n)
            
    elif ham_type == 'XY':
        A = -1/2 * ((1 + gam) / 2 * np.kron(sX, sX) + (1 - gam) / 2 
                     * np.kron(sY, sY) + lam * np.kron(sI, sZ))
        for k in range(num_hams):
            hloc[k] = A.reshape((d,)*2*n)
            
    elif ham_type == 'Ising-G':
        A = -np.kron(sX, sX) + 0.5 * (np.kron(sZ, sI) + np.kron(sI, sZ))
        for k in range(num_hams):
            hloc[k] = A.reshape((d,)*2*n)
            
    elif ham_type == 'XX-G':
        A = np.kron(sX, sX) + np.kron(sY, sY)
        for k in range(num_hams):
            hloc[k] = np.real(A.reshape((d,)*2*n))
            
    else:
        print('Error: Invalid Hamiltonian entered')
        return None
    
    return hloc


