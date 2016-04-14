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
#include "decoder.cpp"
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
        MatrixXd hidden_activation(const MatrixXd & v_state,const MatrixXd & l_state);
        MatrixXd visible_activation(const MatrixXd & h_state);
        MatrixXd label_activation(const MatrixXd & h_state);
        
        MatrixXd sample_hidden(MTRand & random,const MatrixXd & v_state,const MatrixXd & l_state);
        MatrixXd sample_visible(MTRand & random, const MatrixXd & h_state);
        MatrixXd sample_label(MTRand & random, const MatrixXd & h_state);

        // Core Functions
        void loadParameters(long long int p_index);
        void CD_k(MTRand & random, const MatrixXd& batch_V, const MatrixXd& batch_L);
 
        void train(MTRand & random, const MatrixXd& dataset_V, const MatrixXd& dataset_L);
        vector<double> decode(MTRand & random, Decoder & TC, vector<int>  E, vector<int>  S);
        //void test(MTRand & random,MatrixXd & testset_V, MatrixXd & testset_L,ofstream & output);

        // Utilities
        void saveParameters(long long int p_index);
        void printM(MatrixXd matrix);
        void printM_file(MatrixXd matrix, ofstream & file);
        MatrixXd sigmoid(MatrixXd & matrix); 
        MatrixXd MC_sampling(MTRand & random, MatrixXd & activation);

};

#endif
