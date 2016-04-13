#ifndef CRBM_H
#define CRBM_H

#include <vector>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <stdlib.h>
#include "MersenneTwister.h"
#include <Eigen/Core>

using namespace std;
using namespace Eigen;

class crbm {
    
    public:
        
        // RBM Parameters
        int epochs;
        int batch_size;
        int CD_order;
        int n_h;
        int n_v;
        int n_l;
        double L2_par;
        double learning_rate;
        
        MatrixXd W;
        MatrixXd U;
        VectorXd b;
        VectorXd c;
        VectorXd d;
        
        MatrixXd dU;
        MatrixXd dW;
        VectorXd dB;
        VectorXd dC;
        VectorXd dD;
        
        // Constructor
        crbm(MTRand & random);
        
        // Sample functions
        MatrixXd hidden_activation(MatrixXd & v_state,MatrixXd & l_state);
        MatrixXd visible_activation(MatrixXd & h_state);
        MatrixXd label_activation(MatrixXd & h_state);

        MatrixXd sample_hidden(MTRand & random,MatrixXd & v_state,MatrixXd & l_state);
        MatrixXd sample_visible(MTRand & random, MatrixXd & h_state);
        MatrixXd sample_label(MTRand & random, MatrixXd & h_state);

        // Core Functions
        void CD_k(MTRand & random, MatrixXd &batch_V, MatrixXd & batch_L);
        //void SGD(MTRand & random, MatrixXd &batch_V, MatrixXd & batch_L); 
        //void predict_label(MTRand & random, MatrixXd & batch_V); 
        void accuracy(MTRand & random, MatrixXd & batch_V, MatrixXd & batch_L);
 
        void train(MTRand & random, MatrixXd & dataset_V, MatrixXd & dataset_L);
        void sample(MTRand & random, ofstream & output);
        //void test(MTRand & random,MatrixXd & testset_V, MatrixXd & testset_L,ofstream & output);

        // Utilities
        void printM(MatrixXd matrix);
        void printM_file(MatrixXd matrix, ofstream & file);
        MatrixXd sigmoid(MatrixXd & matrix); 
        MatrixXd MC_sampling(MTRand & random, MatrixXd & activation);

};

#endif
