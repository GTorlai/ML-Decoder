#include <stdlib.h>
#include <time.h>
#include <fstream>
#include "rbm.cpp"

int main(int argc, char* argv[]) {
   
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
    get_option("nL","Number of Label Units",argc,argv,Parameters,Helper);
    get_option("lr","Learning Rate",argc,argv,Parameters,Helper);
    get_option("L2","L2 Regularization Amplitude",argc,argv,Parameters,Helper);
    get_option("CD","Contrastive Divergence",argc,argv,Parameters,Helper);
    get_option("ep","Training Epochs",argc,argv,Parameters,Helper);
    get_option("bs","Batch Size",argc,argv,Parameters,Helper);
    get_option("nH" ,"Number of Hidden Units",argc,argv,Parameters,Helper);
    

    Reg_id = "Weight Decay";
    
    vector<double> results;

    MTRand random(1357);
    
    if (command.compare("train") == 0) {
            
        int train_size = 500000;
        int valid_size = 100;

        vector<MatrixXd> dataset(2);
        MatrixXd train_E(train_size,int(Parameters["nV"]));
        MatrixXd train_S(train_size,int(Parameters["nL"]));
        
        //vector<MatrixXd> validSet(2);
        //MatrixXd valid_E(valid_size,int(Parameters["nV"]));
        //MatrixXd valid_S(valid_size,int(Parameters["nL"]));
 
        dataset = loadDataset(train_size,"Train",Parameters["p"],Parameters);
        train_E = dataset[0];
        train_S = dataset[1];
        //validSet = loadValidSet(Parameters);
        //valid_E = validSet[0];
        //valid_S = validSet[1];
        
        //string validationName = buildObserverName(network,model,Parameters,
        //                                        CD_id, Reg_id); 
 
        //ofstream fout(validationName);

        rbm rbm(random,Parameters, int(Parameters["nV"]),
                                   int(Parameters["nH"]),
                                   int(Parameters["nL"]));
        
        rbm.printNetwork(network);

        string modelName = buildModelName(network,model,Parameters,
                                         CD_id, Reg_id);
        
        int L = int(sqrt(Parameters["nV"]/2));
        Decoder TC(L);
        
        string modelNameONLINE = buildModelName_ONLINE(network,model,Parameters,
                                                CD_id, Reg_id);
         
        rbm.train(random,modelNameONLINE,TC,train_E,train_S);//valid_E,valid_S,fout); 
        rbm.saveParameters(modelName); 
        
    }
    
    if (command.compare("decode") == 0) {

        string set = "Test";
        
        //cout << "Decoding p= " << Parameters["p"] << "   A = ";
 
        int test_size = 100000;
        
        vector<MatrixXd> dataset(2);
        MatrixXd data_E;
        MatrixXd data_S;
        data_E.setZero(test_size,int(Parameters["nV"]));
        data_S.setZero(test_size,int(Parameters["nL"]));
 
        int L = int(sqrt(Parameters["nV"]/2));
        Decoder TC(L);

        string modelName = buildModelName(network,model,Parameters,
                                          CD_id, Reg_id);
        string accuracyName = buildAccuracyName(network,model,Parameters,set,
                                                CD_id, Reg_id); 
        //cout << modelName << endl;
        //ofstream fout(accuracyName);
        
        
        rbm rbm(random,Parameters, int(Parameters["nV"]),
                                   int(Parameters["nH"]),
                                   int(Parameters["nL"]));
            
        rbm.loadParameters(modelName); 
 
        dataset = loadDataset(test_size,"Test",Parameters["p"],Parameters);
        data_E = dataset[0];
        data_S = dataset[1];
        
        //results = rbm.decode(random,TC,data_E,data_S);
        results = rbm.decodeSTAT(random,TC,data_E,data_S);

        //fout << Parameters["p"] << "    ";
        //fout << results[0] << "    "; 
        //fout << results[1] << "    "; 
        //fout << results[2] << "    "; 
        //fout << results[3] << "    ";
        //fout << endl; 
        //fout.close();

    }     

}
