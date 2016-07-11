#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <fstream>
#include "dbn.cpp"

int main(int argc, char* argv[]) {
   
    //setNbThreads(8);

    map<string,string> Helper;
    map<string,float> Parameters;
    string CD_id;
    string Reg_id;

    
    initializeParameters(Parameters); 
    string model = "TC2d";
    string command = argv[1];
    string network = argv[2];
     
    get_option("p","Error Probability",argc,argv,Parameters,Helper);
    get_option("nV","Number of Visible Units",argc,argv,Parameters,Helper);
    get_option("l" ,"Number of Hidden Layers",argc,argv,Parameters,Helper); 
    get_option("nL","Number of Label Units",argc,argv,Parameters,Helper);
    get_option("lr","Learning Rate",argc,argv,Parameters,Helper);
    get_option("L2","L2 Regularization Amplitude",argc,argv,Parameters,Helper);
    get_option("CD","Contrastive Divergence",argc,argv,Parameters,Helper);
    //get_option("PCD","Persistent Contrastive Divergence",argc,argv,Parameters,Helper);
    get_option("ep","Training Epochs",argc,argv,Parameters,Helper);
    get_option("bs","Batch Size",argc,argv,Parameters,Helper);
    get_option("p_drop","Dropout Probability",argc,argv,Parameters,Helper);
    get_option("beta","Mixing rate of likelihoods",argc,argv,Parameters,Helper);

    //if (network.compare("DBN") == 0) {
    //    
    //    for (int l=1; l<Parameters["l"]+1; ++l) {
    //    
    //    string hid = "nH" + boost::str(boost::format("%d") % l);
    //    string hid_helper = "Number of Hidden Units on Layer ";
    //    hid_helper += boost::str(boost::format("%d") % l); 
    //    get_option(hid,  hid_helper ,argc,argv,Parameters,Helper);
    //    }
    //}

    //else {
    Parameters["l"] = 1;
    get_option("nH" ,"Number of Hidden Units",argc,argv,Parameters,Helper);
    //}

    //if (int(Parameters["PCD"]) > 0) CD_id = "persistent";
    //else CD_id = "default";
    if (Parameters["p_drop"] > 0) Reg_id = "Dropout";
    else Reg_id = "Weight Decay";

    MTRand random(1357);
    
    //clock_t begin = clock();
    
    if (command.compare("train") == 0) {
            
        int train_size = 200000;

        vector<MatrixXd> dataset(2);
        MatrixXd train_E(train_size,int(Parameters["nV"]));
        MatrixXd train_S(train_size,int(Parameters["nL"]));
 
        dataset = loadDataset(train_size,"Train",Parameters);
        train_E = dataset[0];
        train_S = dataset[1];
            
        crbm crbm(random,Parameters, int(Parameters["nV"]),
                                     int(Parameters["nH"]),
                                     int(Parameters["nL"]));
        
        crbm.printNetwork(network);

        string modelName = buildModelName(network,model,Parameters,
                                          CD_id, Reg_id);
        
        crbm.train(random,network,train_E,train_S); 
        crbm.saveParameters(modelName); 
        
    }
    
    if (command.compare("decode") == 0) {

        int size = 10000;   
        string set = "Test";

        vector<MatrixXd> dataset(2);
        MatrixXd data_E(size,int(Parameters["nV"]));
        MatrixXd data_S(size,int(Parameters["nL"]));
        
                int L = int(sqrt(Parameters["nV"]/2));
        Decoder TC(L);

        string modelName = buildModelName(network,model,Parameters,
                                          CD_id, Reg_id);
        string accuracyName = buildAccuracyName(network,model,Parameters,set,
                                                CD_id, Reg_id); 
        
        ofstream fout(accuracyName);
        
        double accuracy;

        crbm crbm(random,Parameters, int(Parameters["nV"]),
                                         int(Parameters["nH"]),
                                         int(Parameters["nL"]));
            
        crbm.loadParameters(modelName); 
        
        dataset = loadDataset(10000,"Test",Parameters);
        data_E = dataset[0];
        data_S = dataset[1];
    
        accuracy = crbm.decode(random,TC,data_E,data_S);
        
        fout << Parameters["p"] << "    ";
        fout << accuracy << "    "; 
       
        dataset.clear();

        dataset = loadDataset(10000,"Train",Parameters);
        data_E = dataset[0];
        data_S = dataset[1];
    
        accuracy = crbm.decode(random,TC,data_E,data_S);
        
        fout << accuracy << endl; 
  
        fout.close();

    }     
     
}
