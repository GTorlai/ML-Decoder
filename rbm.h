#ifndef RBM_H
#define RBM_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "decoder.cpp"

using namespace std;
using namespace Eigen;

class rbm {
    
    public:
        
        // RBM Parameters
        int epochs;
        int batch_size;
        int CD_order;
        int n_h;
        int n_v;
        double L2_par;
        double learning_rate;
        
        MatrixXd W;
        VectorXd b;
        VectorXd c;

        MatrixXd dW;
        VectorXd dB;
        VectorXd dC;
        
        // Constructor
        rbm(MTRand & random, map<string,float>& parameters,int nV, int nH);
        
        // Sample functions
        MatrixXd hidden_activation(MatrixXd v_state);
        MatrixXd visible_activation(MatrixXd h_state);

        MatrixXd sample_hidden(MTRand & random,MatrixXd v_state);
        MatrixXd sample_visible(MTRand & random, MatrixXd h_state);
        
        // Core Functions
        void CD_k(MTRand & random, MatrixXd batch); 
        void train(MTRand & random, MatrixXd dataset);
        void sample(MTRand & random, ofstream & output);
        void training_sweep(MTRand & random, MatrixXd batch);
 
        // Utilities
        void saveParameters(string& modelName);
        void loadParameters(string& modelName);
        double reconstruction_error(MatrixXd data, MatrixXd h_state);
        MatrixXd sigmoid(MatrixXd matrix); 
        MatrixXd MC_sampling(MTRand & random, MatrixXd activation);
        void printNetwork();

};

#endif
