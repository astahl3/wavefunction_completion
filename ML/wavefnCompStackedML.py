'''
Sample supervised learning script for wavefunction completion using stacked
networks of fully connected layers, where the networks are arranged analogous
to the ramping up of bond dimension performed in tensor completion for 
wavefunctions.

Author: Aaron Stahl (2024)
'''

import tensorflow as tf
import numpy as np
#import matplotlib as mpl
from genEigenstates import genEigenstates
import compHelperFunctions as hf
from genLocalHams import genLocalHams
from applyHam import applyHam
from scipy.sparse.linalg import eigsh, LinearOperator
import random 
from wavefunctionCompEx import doCompletion

# Ensure TensorFlow 2.x is used
print(tf.__version__)

# Wavefunction parameters
ham_type = 'rand-homog-r' # type of Hamiltonian (options in genLocalHams.py)
N = 14 # number of lattice sites
d = 2 # local dimension of each lattice site
n = 2 # interaction length of local operators
use_pbc = False # periodic boundary conditions
gam = 1 # for model="Ising" (see genLocalHams.py)
lam = 1 # for model="Ising" (see genLocalHams.py)
Enum = 1 # desired eigenstate (1 = ground state, 2 = first excited, ...)
sample_rate = 0.80

# ML parameters
num_iters = 5000  # Number of training iterations
num_test_iters = 200  # Number of testing iterations
batch_size = 200
networks = []
data_loc = '/Users/astahl/repositories/wavefunction_completion/ML_training_data/N14_d2_n2_OBC_s80_t1000_fixedLocs.npz'

# Use nuclear norm as loss function across central bipartition
def nuclear_norm_loss(y_true, y_pred):
    # Reshape the predicted and true wavefunctions into matrices
    # For example, splitting the state into two equal parts
    # Assuming y_pred is a flat tensor of shape [batch_size, d**N]
    size = int(tf.math.sqrt(tf.cast(tf.shape(y_pred)[1], tf.float32)))
    y_pred_reshaped = tf.reshape(y_pred, [-1, size, size])
    y_true_reshaped = tf.reshape(y_true, [-1, size, size])
    
    # Compute singular values for each matrix in the batch
    s_pred = tf.linalg.svd(y_pred_reshaped, compute_uv=False)
    s_true = tf.linalg.svd(y_true_reshaped, compute_uv=False)
    
    # Sum of singular values (nuclear norm)
    nuclear_norm_pred = tf.reduce_sum(s_pred, axis=1)
    nuclear_norm_true = tf.reduce_sum(s_true, axis=1)
    
    # Minimize the difference in nuclear norms between predictions and true values
    return tf.reduce_mean(tf.abs(nuclear_norm_pred - nuclear_norm_true))

def masked_mse(y_true, y_pred, psi_mask):
    # Calculate MSE loss for each element and apply mask
    error = tf.square(y_true - y_pred)  # MSE
    masked_error = tf.multiply(error, psi_mask)  # Apply mask
    return tf.reduce_mean(masked_error)  # Average over non-masked elements only

def build_model(num_nodes, num_layers=10):
    model = tf.keras.Sequential()
    # Start with a smaller number of nodes
    model.add(tf.keras.layers.Input(shape=(16384,)))  # Input layer corresponding to 96 elements at chi=2

    # Gradual increase in the number of nodes
    # Increase roughly mimicking some key bond dimensions
    for k in range(num_layers):
        model.add(tf.keras.layers.Dense(num_nodes, activation='relu'))
    
    # Adding layers corresponding to increasing bond dimension
    #for nodes in nodes_in_layers:
    #    model.add(tf.keras.layers.Dense(nodes, activation='relu'))
    #    model.add(tf.keras.layers.Dropout(0.2))  # Adding some dropout for regularization

    # Final layer
    model.add(tf.keras.layers.Dense(d**N))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    #model.compile(optimizer='adam', loss=lambda y_true, y_pred: masked_mse(y_true, y_pred, psi_mask))
    return model


# Generate batch data for training the network
def generate_data(num_samples):
    states = {}
    incomplete_states = {}
    for i in range(num_samples):
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
            psi = psi[:,Enum]
            
        # Initialize incomplete state
        psi_inc, slocs = hf.genIncompleteState(psi, sample_rate)
        all_locs = np.arange(psi.size)
        ulocs = np.setdiff1d(all_locs, slocs)
        
        states[i] = psi.reshape(d**N,)
        incomplete_states[i] = psi_inc.reshape(d**N,)
        #psi, _ = genEigenstates(1, N, Enum, ham_type, use_pbc, d, n)  # Assuming genEigenstates returns the complete state psi
        #psi_inc, slocs = hf.genIncompleteState(psi, sample_rate)  # Your method to create incomplete states
        #states[i] = psi
        #incomplete_states[i] = psi_inc
    
    print('Finished generating {num_samples} eigenstates for testing')
    return incomplete_states, states

# Neural Network Model

# Load data (if you've already run the above)
with np.load(data_loc) as data:
        train_incomplete = data['train_incomplete']
        train_complete = data['train_complete']
        test_incomplete = data['test_incomplete']
        test_complete = data['test_complete']
        psi_mask = data['psi_mask']
        
slocs = np.where(psi_mask == 0)[0]
        
# DEBUGGING - PICKING RANDOM STATE TO CONFIRM IT IS COMPLETABLE
'''
psi_seed = random.randint(1,train_incomplete.shape[0])
psi_test = train_complete[psi_seed,:]
psi_test_inc = train_incomplete[psi_seed,:]
ulocs = np.where(psi_mask == 1)[0]
slocs = np.where(psi_mask == 0)[0]
psi_test_comp, fid_err = doCompletion(psi_test, psi_test_inc, ulocs, N, d, n, use_pbc)
'''

# Creating networks with increasing capacity
nodes_in_layers = [96, 204, 352, 512, 704, 928, 1184, 1680, 1964, 2272, 2604, 2960]
epochs = [100, 50, 50, 50, 50, 20, 20, 20, 20, 20, 20, 20]
num_networks = len(nodes_in_layers)
for i in range(num_networks):
    net = build_model(nodes_in_layers[i-1])
    networks.append(net)   

# Training each network sequentially
for i in range(len(networks)):
    if i == 0:
        networks[i].fit(train_incomplete, train_complete, epochs=epochs[i], batch_size=batch_size, validation_split=0.1)
    else:
        # Apply corrections based on the previous network
        predictions = networks[i-1].predict(train_incomplete)
        predictions[:,slocs] = train_complete[:,slocs]
        net.fit(predictions, train_complete, epochs=epochs[i], batch_size=batch_size, validation_split=0.1)

# Evaluating the model
predictions = networks[0].predict(test_incomplete)
predictions[:,slocs] = test_complete[:,slocs]
for i in range(1,len(networks)):
    predictions = networks[i].predict(predictions)
    predictions[:,slocs] = test_complete[:,slocs]
    
for i in range(test_complete.shape[0]):
    fidelity = hf.stateFidelity(predictions[i,:], test_complete[i,:])
    print(f"Fidelity for test case {i}: {fidelity}")

# Save result in HDF5 format
for k in range(0, len(networks)):
    save_name = 'net' + str(k) + '_' + str(nodes_in_layers[k]) + 'nodes.h5'
    save_dir = '/Users/astahl/repositories/wavefunction_completion/ML_models/fully_conn_network_stack/'
    save_loc = save_dir + save_name
    networks[k].save(save_loc)

#model.save('/Users/astahl/repositories/wavefunction_completion/ML_models/compML0.h5')  # Saves the entire model in HDF5 format

