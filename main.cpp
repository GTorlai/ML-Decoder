#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <fstream>
#include "rbm.cpp"

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
    get_option("ep","Training Epochs",argc,argv,Parameters,Helper);
    get_option("bs","Batch Size",argc,argv,Parameters,Helper);

    Parameters["l"] = 1;
    get_option("nH" ,"Number of Hidden Units",argc,argv,Parameters,Helper);

    Reg_id = "Weight Decay";

    MTRand random(1357);
    
    //clock_t begin = clock();
           
    if (command.compare("train") == 0) {
            
        int train_size = 200000;
        int valid_size = 100;

        vector<MatrixXd> dataset(2);
        MatrixXd train_E(train_size,int(Parameters["nV"]));
        MatrixXd train_S(train_size,int(Parameters["nL"]));
        
        vector<MatrixXd> validSet(2);
        MatrixXd valid_E(valid_size,int(Parameters["nV"]));
        MatrixXd valid_S(valid_size,int(Parameters["nL"]));
 
        dataset = loadDataset(train_size,"Train",Parameters);
        train_E = dataset[0];
        train_S = dataset[1];
        validSet = loadValidSet(Parameters);
        valid_E = validSet[0];
        valid_S = validSet[1];
        
        string validationName = buildObserverName(network,model,Parameters,
                                                CD_id, Reg_id); 
 
        ofstream fout(validationName);

        rbm rbm(random,Parameters, int(Parameters["nV"]),
                                   int(Parameters["nH"]),
                                   int(Parameters["nL"]));
        
        rbm.printNetwork(network);

        string modelName = buildModelName(network,model,Parameters,
                                          CD_id, Reg_id);
        
        int L = int(sqrt(Parameters["nV"]/2));
        Decoder TC(L);
 
        rbm.train(random,network,TC,train_E,train_S,valid_E,valid_S,fout); 
        rbm.saveParameters(modelName); 
        
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

        rbm rbm(random,Parameters, int(Parameters["nV"]),
                                         int(Parameters["nH"]),
                                         int(Parameters["nL"]));
            
        rbm.loadParameters(modelName); 
        
        dataset = loadDataset(10000,"Test",Parameters);
        data_E = dataset[0];
        data_S = dataset[1];
    
        accuracy = rbm.decode(random,TC,data_E,data_S);
        
        fout << Parameters["p"] << "    ";
        fout << accuracy << "    "; 
       
        dataset.clear();

        dataset = loadDataset(10000,"Train",Parameters);
        data_E = dataset[0];
        data_S = dataset[1];
    
        accuracy = rbm.decode(random,TC,data_E,data_S);
        
        fout << accuracy << endl; 
  
        fout.close();

    }     
     
}
