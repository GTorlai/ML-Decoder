#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <fstream>
#include "dbn.cpp"

int main(int argc, char* argv[]) {
   
    //setNbThreads(8);

    map<string,string> Helper;
    map<string,float> Parameters;
    
    string model = "TC2d";
    string command = argv[1];
    string network = argv[2];
     
    get_option("p","Error Probability",argc,argv,Parameters,Helper);
    get_option("nV","Number of Visible Units",argc,argv,Parameters,Helper);
    get_option("l" ,"Number of Hidden Layers",argc,argv,Parameters,Helper); 

    if (network.compare("DBN") == 0) {
        
        for (int l=1; l<Parameters["l"]+1; ++l) {
        
        string hid = "nH" + boost::str(boost::format("%d") % l);
        string hid_helper = "Number of Hidden Units on Layer ";
        hid_helper += boost::str(boost::format("%d") % l); 
        get_option(hid,  hid_helper ,argc,argv,Parameters,Helper);
        }
    }

    else {
        Parameters["l"] = 1;
        get_option("nH" ,"Number of Hidden Units",argc,argv,Parameters,Helper);
    }

    get_option("nL","Number of Label Units",argc,argv,Parameters,Helper);
    get_option("lr","Learning Rate",argc,argv,Parameters,Helper);
    get_option("L2","L2 Regularization Amplitude",argc,argv,Parameters,Helper);
    get_option("CD","Contrastive Divergence",argc,argv,Parameters,Helper);
    get_option("ep","Training Epochs",argc,argv,Parameters,Helper);
    get_option("bs","Batch Size",argc,argv,Parameters,Helper);

    MTRand random(1234);
    
    //clock_t begin = clock();
 
    if (command.compare("train") == 0) {
        
        
        int size = 200000;

        vector<MatrixXd> dataset(2);
        MatrixXd data_E(size,int(Parameters["nV"]));
        MatrixXd data_S(size,int(Parameters["nL"]));
        
        dataset = loadDataset(size,"Train",Parameters);
        data_E = dataset[0];
        data_S = dataset[1];
        if (network.compare("CRBM") == 0) {
            
            crbm crbm(random,Parameters, int(Parameters["nV"]),
                                         int(Parameters["nH"]),
                                         int(Parameters["nL"]));

            //crbm.printNetwork();
            string modelName = buildModelName(network,model,Parameters);
            crbm.train(random,data_E,data_S); 
            crbm.saveParameters(modelName); 
        }
        if (network.compare("DBN") == 0) {

            string modelName = buildModelName(network,model,Parameters);
            dbn dbn(random,Parameters);
            dbn.Train(random,data_E,data_S);
            dbn.saveParameters(modelName);
        } 
    }
    
    if (command.compare("decode") == 0) {
        
        int size = 10000;
        int n_measure = 1000;
        
        vector<int> E0;
        vector<int> S0;
        E0.assign(int(Parameters["nV"]),0);
        S0.assign(int(Parameters["nL"]),0);
        vector<double> accuracy;
        accuracy.assign(2,0);
        double E_percent = 0.0;
        double S_percent = 0.0;
 
        vector<MatrixXd> dataset(2);
        MatrixXd data_E(size,int(Parameters["nV"]));
        MatrixXd data_S(size,int(Parameters["nL"]));
        
        dataset = loadDataset(size,"Test",Parameters);
        data_E = dataset[0];
        data_S = dataset[1];
        int L = int(sqrt(Parameters["nV"]/2));

        Decoder TC(L);

        string modelName = buildModelName(network,model,Parameters);
        
        string accuracyName = buildAccuracyName(network,model,Parameters); 
        
        ofstream fout(accuracyName);
 
        if (network.compare("CRBM") == 0) {
            
            crbm crbm(random,Parameters, int(Parameters["nV"]),
                                         int(Parameters["nH"]),
                                         int(Parameters["nL"]));
            
            crbm.loadParameters(modelName); 
 
            for (int n=0; n<n_measure; n++) {
            
                //cout << n << endl;
                
                for (int k=0; k<int(Parameters["nL"]); k++) {
                
                    S0[k] = data_S(n,k);
                }

                for (int j=0; j<int(Parameters["nV"]); j++) {
                    
                    E0[j] = data_E(n,j);
                }

                accuracy = crbm.decode(random,TC,E0,S0);
                
                S_percent += accuracy[0];
                E_percent += accuracy[1];    
        
            }
        
        }
 
        else if (network.compare("DBN") == 0) {
            
            dbn dbn(random,Parameters);
            
            dbn.loadParameters(modelName); 
            
            for (int n=0; n<n_measure; n++) {
            
                //cout << n << endl;
                
                for (int k=0; k<int(Parameters["nL"]); k++) {
                
                    S0[k] = data_S(n,k);
                }

                for (int j=0; j<int(Parameters["nV"]); j++) {
                    
                    E0[j] = data_E(n,j);
                }

                accuracy = dbn.decode(random,TC,E0,S0);
                
                S_percent += accuracy[0];
                E_percent += accuracy[1];    
            }
        }
        //
        S_percent /= n_measure;
        E_percent /= n_measure;
        //cout << endl << endl << endl;
        //cout << "Syndrome Accuracy: " << S_percent << "%"<< endl;
        //cout << "Correction Accuracy: " << E_percent << "%" << endl;
 
        fout << Parameters["p"] << "  " << "S   " << "E" << endl;
        fout << S_percent << "    " << E_percent << endl; 
        
        fout.close();
    }
    //clock_t end = clock();
    //double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    //cout << "Elementwise elapse time: " << elapsed_secs << endl << endl;

     
}
