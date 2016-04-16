#ifndef UTILITIES_H
#define UTITLITES_H

#include <Eigen/Core>
#include <sstream>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <map>
#include <boost/format.hpp>

using namespace std;

//*****************************************************************************
// Get command line options
//*****************************************************************************

void get_option(const string& arg, const string& description,
                int argc, char** argv, 
                map<string,float>& par, map<string,string>& helper)
{
    string flag = "--" + arg ;
    for (int i=3; i<argc; ++i) {
            
        if (flag.compare(argv[i]) ==0) {

            par[arg] = atof(argv[i+1]);
            break;
        }
    }

    helper[arg] = description;
}


//*****************************************************************************
// Generate the Base Name of the simulation
//*****************************************************************************

string buildBaseName(const string& network, const string& model,
                     map<string,float>& par) 
{

    string baseName = network;
    baseName += "_CD";
    baseName += boost::str(boost::format("%.0f") % par["CD"]);
    baseName += "_nH";
    if (network.compare("DBN") == 0) {
        
        for (int l=0; l<par["l"]; ++l) {
            string hid = "nH" + to_string(l);
            baseName += boost::str(boost::format("%.0f") % par[hid]);
            baseName += "_";
        }
    }
    else {
        baseName += boost::str(boost::format("%.0f") % par["nH"]);
    }
    baseName += "_bs";
    baseName += boost::str(boost::format("%.0f") % par["bs"]);
    baseName += "_ep";
    baseName += boost::str(boost::format("%.0f") % par["ep"]);
    baseName += "_lr";
    baseName += boost::str(boost::format("%.3f") % par["lr"]);
    baseName += "_L2Reg";
    baseName += boost::str(boost::format("%.3f") % par["L2"]);
    baseName += "_";
    baseName += model;
    baseName += "_L";
    int L = int(sqrt(par["nV"]/2));
    baseName += boost::str(boost::format("%d") % L);
    
    return baseName;
}


//*****************************************************************************
// Load datasets 
//*****************************************************************************

vector<Eigen::MatrixXd> loadDataset(int size, map<string,float>& parameters) 

{
    string sSize = to_string(size/1000);
    Eigen::MatrixXd data_E(size,int(parameters["nV"]));
    Eigen::MatrixXd data_S(size,int(parameters["nL"]));
    
    vector<Eigen::MatrixXd> dataset;

    string baseName     = "data/datasets/train/L4/";
    string errorName    = baseName + "E_Train_";
    string syndromeName = baseName + "S_Train_";
    
    errorName    += sSize;
    syndromeName += sSize;
    errorName    += "k_p";
    syndromeName += "k_p";
    errorName    += boost::str(boost::format("%.3f") % parameters["p"]);
    syndromeName += boost::str(boost::format("%.3f") % parameters["p"]);
    errorName    += ".txt";
    syndromeName += ".txt";

    ifstream dataFile_E(errorName);
    ifstream dataFile_S(syndromeName);
    
    for (int n=0; n<size; n++) {
        
        for (int j=0; j<int(parameters["nV"]); j++) {

            dataFile_E >> data_E(n,j);
        }

        for (int k=0; k<int(parameters["nL"]); k++) {

            dataFile_S >> data_S(n,k);
        }
    }

    dataset.push_back(data_E);
    dataset.push_back(data_S);

    return dataset;
}




//*****************************************************************************
// Print Matrix or Vector on the screen
//*****************************************************************************

template<typename T> 
ostream& operator<< (ostream& out, const Eigen::MatrixBase<T>& M)
{    
    for (size_t i =0; i< M.rows(); ++i) {
        
        for (size_t j =0; j< M.cols(); ++j) {
            
            out << M(i,j)<< " ";
        }
        
        if (M.cols() > 1) out << endl;
    }
    
    out << endl;

    return out;
}


//*****************************************************************************
// Write Matrix or Vector on file 
//*****************************************************************************

template<typename T> 
void write (ofstream& fout,const Eigen::MatrixBase<T>& M)
{
    for (size_t i =0; i< M.rows(); ++i) {
        
        for (size_t j =0; j< M.cols(); ++j) {
            
            fout << M(i,j)<< " ";
        }
        
        if (M.cols() > 1) fout << endl;
    }
    
    fout << endl;
}


//*****************************************************************************
// Apply sigmoid function to an array 
//*****************************************************************************

template<typename T> Eigen::ArrayXXd sigmoidTEMP(const Eigen::ArrayBase<T>& M)
{
    return M.exp()/(1.0+M.exp());

}


#endif
