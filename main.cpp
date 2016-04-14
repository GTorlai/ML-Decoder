#include "rbm.cpp"
#include "crbm.cpp"
#include <stdlib.h>
#include <string.h>
#include <sstream>
#include <vector>
#include <time.h>
#include <omp.h>
#include <fstream>
#include <iostream>

int main(int argc, char* argv[]) {
    
    long long int p_index;

    for (int i=1;i<argc;i++) {
        
        if (strcmp(argv[i],"--p") == 0) {
           p_index = atoi(argv[i+1]);
        }
    } 


    MTRand random(1234);

    Decoder TC(4);
    
    double p;

    p = p_index/100.0;

    //string extension = "_Train_200k_p";
    //if (p_index<10) {
    //    extension += "0";
    //}
    //extension += to_string(p_index);
    //extension += ".txt";
    //string nameE = "data/datasets/train/E" + extension;
    //string nameS = "data/datasets/train/S" + extension;
    //
    //ofstream outputE(nameE);
    //ofstream outputS(nameS);
    //ofstream outputE("E_Train_10k_p05.txt");
    //ofstream outputS("S_Train_10k_p05.txt");

    crbm crbm(random);
     
    //TC.generateDataset(random,p,outputS,outputE);

    //clock_t begin = clock();
    
    
    string dataName_E = "data/datasets/test/E_Test_10k_p";
    string dataName_S = "data/datasets/test/S_Test_10k_p";
    if (p_index < 10) {
        dataName_E += "0";
        dataName_S += "0";
    }
    dataName_E += to_string(p_index);
    dataName_S += to_string(p_index);
    dataName_E += ".txt";
    dataName_S += ".txt";

    //
    ifstream dataFile_E(dataName_E);
    ifstream dataFile_S(dataName_S);

    int size = 1000;

    MatrixXd data_E(size,crbm.n_v);
    MatrixXd data_S(size,crbm.n_l);
    
    cout << "\n\n Loading data...\n\n";
    
    for (int n=0; n<size; n++) {
        for (int j=0; j<crbm.n_v; j++) {

            dataFile_E >> data_E(n,j);
        }
        for (int k=0; k<crbm.n_l; k++) {

            dataFile_S >> data_S(n,k);
        }
    }

    vector<int> E0;
    vector<int> S0;
    E0.assign(crbm.n_v,0);
    S0.assign(crbm.n_l,0);
    vector<double> accuracy;
    accuracy.assign(2,0);
    double E_percent = 0.0;
    double S_percent = 0.0;

    crbm.loadParameters(p_index); 
    string outName = "data/Accuracy_p";
    if (p_index < 10) {
        outName += "0";
    }
    outName += to_string(p_index);
    outName += ".txt";
 
    ofstream fout(outName);
 
    int n_measure = 200;
    clock_t begin = clock();

    for (int n=0; n<n_measure; n++) {
	cout << n << endl;
        for (int k=0; k<crbm.n_l; k++) {
        
            S0[k] = data_S(n,k);
        }

        for (int j=0; j<crbm.n_v; j++) {
            
            E0[j] = data_E(n,j);
        }


        accuracy = crbm.decode(random,TC,E0,S0);
        S_percent += accuracy[0];
        
        E_percent += accuracy[1];    
    }

    S_percent /= n_measure;
    E_percent /= n_measure;
    fout << p << "  " << "S   " << "E" << endl;
    fout << S_percent << "    " << E_percent << endl; 
    fout.close();
    cout << endl << endl << endl;
    cout << "Syndrome Accuracy: " << S_percent << endl;
    cout << "Correction Accuracy: " << E_percent << endl;
    
    
    
    //cout << "\n\n Training... \n\n";

    //crbm.train(random,data_E,data_S); 
    //crbm.saveParameters(p_index); 
    
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Elementwise elapse time: " << elapsed_secs << endl << endl;
 
    
}
