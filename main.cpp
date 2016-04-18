#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <fstream>
#include "dbn.cpp"

int main(int argc, char* argv[]) {
   
    setNbThreads(8);

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
    
    //clock_t end = clock();
    //double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    //cout << "Elementwise elapse time: " << elapsed_secs << endl << endl;

     
}
