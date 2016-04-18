#ifndef DBN_H
#define DBN_H

//#include <vector>
//#include <fstream>
//#include <stdio.h>
//#include <math.h>
//#include <iostream>
//#include <stdlib.h>
//#include <Eigen/Core>
//#include "MersenneTwister.h"
//#include "rbm.cpp"
#include "crbm.cpp"

using namespace std;
using namespace Eigen;

class dbn {
    
    public:
        
        int layers;
        int batch_size;
        
        vector<int> hidden;
        
        vector<rbm> rbms;
        vector<crbm> crbms;

        dbn(MTRand & random,map<string,float>& Parameters); 
        
        MatrixXd propagateDataset(int newLayer, const MatrixXd& dataset);

        MatrixXd topDownInference(MTRand& random,MatrixXd& hTop);
        MatrixXd equilibrateTop(MTRand& random, int steps, MatrixXd& h_state);
        
        void Train(MTRand& random, const MatrixXd& data_E, 
                                   const MatrixXd& data_S);

        void saveParameters(string& modelName);
        void loadParameters(string& modelName);

};

#endif
