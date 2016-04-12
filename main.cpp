#include "rbm.cpp"
#include <stdlib.h>
#include <string.h>
#include <sstream>
#include <vector>
#include <time.h>

int main(int argc, char* argv[]) {
    
    //clock_t begin = clock();
    
    int T;
    int batch_size = 100;
    int epochs = 2000;
    int CD_order = 20;
    double learning_rate;
    double L2;

    for (int i=1;i<argc;i++) {
        
        if (strcmp(argv[i],"--T") == 0) {
           T = atoi(argv[i+1]);
        }
    } 

    string datasetName = "data/datasets/MC_Ising2d_L4_T";
    string samplesName = "data/samples/RBM_samples_T";
    
    if (T<10) {
        datasetName += '0';
        samplesName += '0';
    }

    stringstream ss;
    ss << T;
    string T_string = ss.str();
    
    datasetName += T_string;
    datasetName += ".txt";
    samplesName += T_string;
    samplesName += ".txt";
 
    MTRand random;

    rbm rbm(random);    

    ifstream datasetFile(datasetName);
    ofstream samplesFile(samplesName);

    int size = 100000;

    MatrixXd dataset(size,rbm.n_v);

    for (int n=0; n<size; n++) {
        for (int j=0; j<rbm.n_v; j++) {

            datasetFile >> dataset(n,j);
        }
    }
    //cout << "\n\n TRAINING \n\n";

    rbm.train(random,dataset); 
    
    //clock_t end = clock();
    //double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    //cout << "Elementwise elapse time: " << elapsed_secs << endl << endl;
 
    rbm.batch_size = 1;

    cout << "\n\n SAMPLING \n\n";

    rbm.sample(random,samplesFile);
    
}
