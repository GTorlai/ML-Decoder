//#include "rbm.cpp"
//#include "crbm.cpp"
//#include <string.h>
//#include <sstream>
//#include <vector>
//#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <fstream>
#include <boost/format.hpp>
#include "decoder.cpp"

int main(int argc, char* argv[]) {
    
    float p;
    int size;
    int L;
    int seed;
    string id; 
    
    id = argv[1];

    for (int i=2; i<argc; i++) {

        if (strcmp(argv[i],"--L") == 0) {
            L = atoi(argv[i+1]);
        }
        
        if (strcmp(argv[i],"--p") == 0) {
            p = atof(argv[i+1]);
        }
        
        if (strcmp(argv[i],"--size") == 0) {
            size = atoi(argv[i+1]);
        }
        
        if (strcmp(argv[i],"--seed") == 0) {
            seed = atoi(argv[i+1]);
        }
    }
    
    string baseName = "data/datasets/";
    baseName += id;
    baseName += "/L";
    baseName += to_string(L);
    baseName += "/";
    string extension = "_"; 
    extension += id;
    extension += "_L";
    extension += to_string(L);
    extension += "_";
    extension += to_string(size/1000);
    extension += "k_p";
    extension += boost::str(boost::format("%.3f") % p); 
    extension += ".txt";

    string errorName    = baseName + "Error" + extension;
    string syndromeName = baseName + "Syndrome" + extension;
        
    ofstream file_E(errorName);
    ofstream file_S(syndromeName);

    MTRand random(seed);
    
    Decoder TC(L);
    
    vector<int> E;
    vector<int> S;
    
    cout << "\nGenerating ";
    cout << id;
    cout << " dataset at p = ";
    cout << p;
    cout << endl;

    for (int i=0; i<size; ++i) {
        
        E = TC.generateError(random,p);
        S = TC.getSyndrome(E);
        
        for (int j=0; j<E.size(); ++j) {

            file_E << E[j] << " ";
        }

        file_E << endl;
        
        for (int j=0; j<S.size(); ++j) {

            file_S << S[j] << " ";
        }

        file_S << endl;
    }    
    
    file_E.close();
    file_S.close();
        
}
