#ifndef CRBM_H
#define CRBM_H

#include "rbm.cpp"

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
        float L2_par;
        float learning_rate;
        float alpha;
        string CD_type;

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

        MatrixXd Persistent;
        
        // Constructor
        crbm(MTRand & random, map<string,float>& parameters,
             int nV,int nH,int nL);
        
        // Sample functions
        MatrixXd hidden_activation(const MatrixXd & v_state,
                                   const MatrixXd & l_state);
        MatrixXd visible_activation(const MatrixXd & h_state);
        MatrixXd label_activation(const MatrixXd & h_state);
        
        MatrixXd sample_hidden(MTRand & random,const MatrixXd & v_state,
                                               const MatrixXd & l_state);
        MatrixXd sample_visible(MTRand & random, const MatrixXd & h_state);
        MatrixXd sample_label(MTRand & random, const MatrixXd & h_state);

        // Core Functions
        void CD(MTRand & random, const MatrixXd& batch_V, 
                                 const MatrixXd& batch_L);
        
        void train(MTRand & random, const MatrixXd& dataset_V, 
                                               const MatrixXd& dataset_L);
        
        double decode(MTRand & random, Decoder & TC, 
                        MatrixXd& testSet_E, MatrixXd& testSet_S);

        // Utilities
        void loadParameters(string& modelName);
        void saveParameters(string& modelName);
        void printNetwork();
        MatrixXd sigmoid(MatrixXd & matrix); 
        MatrixXd MC_sampling(MTRand & random, MatrixXd & activation);

};

#endif
