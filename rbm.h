#ifndef RBM_H
#define RBM_H

#include "ToricCode.cpp"

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
        int n_l;
        float L2_par;
        float learning_rate;
        float alpha;
        float beta;
        float p_drop;
        double sparse_par;
        string CD_type;
        string regularization;

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
        rbm(MTRand & random, map<string,float>& parameters,
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
        
        vector<double> validate(MTRand & random, Decoder & TC,
                const MatrixXd& validSet_E, const MatrixXd& validSet_S);
        
        void train(MTRand & random, const string& netName, Decoder & TC,
                const MatrixXd& dataset_V, const MatrixXd& dataset_L);
                //const MatrixXd& validSet_E, const MatrixXd& validSet_S,
                //ofstream & fileName);
        
        vector<double> decode(MTRand & random, Decoder & TC, 
                        MatrixXd& testSet_E, MatrixXd& testSet_S);
        
        vector<double> decodeSTAT(MTRand & random, Decoder & TC, 
                        MatrixXd& testSet_E, MatrixXd& testSet_S);

        double freeEnergy(VectorXd E, VectorXd S);
        // Utilities
        void loadParameters(string& modelName);
        void saveParameters(string& modelName);
        void saveParameters_ONLINE(const string& modelName,int e);
        void loadParameters_ONLINE(const string& modelName,int e);
        void printNetwork(const string& network);
        
        MatrixXd sigmoid(MatrixXd & matrix); 
        MatrixXd MC_sampling(MTRand & random, MatrixXd & activation);

};

#endif
