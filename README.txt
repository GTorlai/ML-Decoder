*******************************************************************************
*******************************************************************************

A Neural Decoder - Code

Written by Giacomo Torlai - September 2016

Language: C++
Libraries: Boost, Eigen

*******************************************************************************
*******************************************************************************

Makefile compiles the two executables:
- datasets.x to generate a dataset
- run.x to train or decode using for the neural network

GENERATING DATASETS

RUNNING THE NEURAL NETWORK

Command: ./run.x COMMAND RBM --flag1 --flag2 ...

The hyper-parameters that must be specified in the command line after RBM are:

--nV : Number of visile units (qubits)
--nL : Number of "label" units (check operators)
--nH : Number of hidden units
--CD : Number of block Gibbs update in the Contrastive Divergence algorithm
--lr : Learning rate for Stochastic Gradient Descent
--ep : Number of training steps
--bs : Batch size 
--L2 : Magnitude of the L2 regularization 
--p  : Error probability for single qubits under dephasing


TRAINING

Command: ./run.x train RBM --flag1 --flag2 ...


DECODING

Command: ./run.x decode RBM --flag1 --flag2 ...

