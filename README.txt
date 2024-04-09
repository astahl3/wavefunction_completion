Wavefunction Completion with Tensor Networks
Author: Aaron Stahl (2024) // aaaron.m.stahl@gmail.com
Author's note: a more comprehensive repository is 
available in Matlab; please email if interested.


OVERVIEW
----------------
This project introduces several new tensor network algorithms 
for reconstructing ("completing") low energy eigenstates of an 
unknown Hamiltonian using a random sample of the wavefunction 
coefficient amplitudes. The completion algorithms leverage 
truncated matrix product states (MPS), randomized tensor tree 
networks (TTN), and other tensor-oriented structures to offer 
powerful tools for wavefunction completion. Starting from only a 
sparse sampling of amplitudes, these routines commonly obtain 
completed states with fidelity values near the limits of numerical 
precision.

CITATION
-------------
This repository is associated with the article, "Reconstruction of 
Randomly Sampled Quantum Wavefunctions using Tensor 
Methods" by Aaron Stahl and Glen Evenbly (2023). For a detailed 
theoretical background and numerical results, please refer to: 

https://arxiv.org/abs/2310.01628

Abstract: We propose and test several tensor network based 
algorithms for reconstructing the ground state of an (unknown) 
local Hamiltonian starting from a random sample of the 
wavefunction amplitudes. These algorithms, which are based on 
completing a wavefunction by minimizing the block Renyi 
entanglement entropy averaged over all local blocks, are 
numerically demonstrated to reliably reconstruct ground states 
of local Hamiltonians on 1-D lattices to high fidelity, often at the 
limit of double-precision numerics, while potentially starting from 
a random sample of only a few percent of the total wavefunction 
amplitudes.

FEATURES
----------------
* Exact diagonalization of local Hamiltonians for calculating 
   eigenvalues and eigenstates
* Wavefunction completion using tensor network methods
* Support for various model options including the critical XX model, 
   Ising model, and randomly generated homogenous and 
   inhomogenous Hamiltonians with arbitrary interaction lengths

INSTALLATION
---------------------
Core functionality included in:
- applyHam.py
- genLocalHams.py
- ncon.py 
- truncatedMPS.py
- allCutSweep.py
- compHelperFunctions.py
- genBlocksTree.py
- oneLayerTree.py
- reverseLayerTree.py

Sample implementations:
- exactDiagEx.py (exact diagonalization)
- wavefunctionCompEx.py (example: MPS and ACS)
- wavefunctionTreeCompEx.py (example: tree tensor network)

ACKNOWLEDGMENTS
--------------------------------
Thank you to Glen Evenbly for his assistance in developing this project.