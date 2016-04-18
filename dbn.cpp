#include "dbn.h"

//***********************************************************************
// Constructor 
//***********************************************************************

dbn::dbn(MTRand & random, map<string,float>& Parameters) 
{
    batch_size = Parameters["bs"];

    layers = Parameters["l"];
    
    rbms.push_back(rbm(random,Parameters,Parameters["nV"],Parameters["nH1"]));
    
    for (int l=2; l<layers; ++l) {
        
        //random.seed();
        //RanGen.push_back(random);
        string vis = "nH" + boost::str(boost::format("%d") % (l-1)); 
        string hid = "nH" + boost::str(boost::format("%d") % l);
        rbms.push_back(rbm(random,Parameters,Parameters[vis],Parameters[hid]));
    }
    
    string vis = "nH" + boost::str(boost::format("%d") % (layers-1)); 
    string hid = "nH" + boost::str(boost::format("%d") % (layers));

    crbms.push_back(crbm(random,Parameters,int(Parameters[vis]),
                                           int(Parameters[hid]),
                                           int(Parameters["nL"])));
    
}


//*****************************************************************************
// Train the network
//***************************************************************************** 

MatrixXd dbn::propagateDataset(int Layer, const MatrixXd& dataset) 
{
    
    MatrixXd newDataset(dataset.rows(),rbms[Layer-1].n_h);
    MatrixXd batch(batch_size,dataset.cols());
    MatrixXd newBatch(batch_size,rbms[Layer-1].n_h);

    int n_batches = dataset.rows()/batch_size;

    for (int b=0; b<n_batches; ++b) {

        batch = dataset.block(b*batch_size,0,batch_size,dataset.cols()); 
        newBatch = rbms[Layer-1].hidden_activation(batch);
        
        for(int s=0; s<batch_size; ++s) {

            newDataset.row(batch_size*b+s) = newBatch.row(s);
        }
    } 

    return newDataset;
}


//*****************************************************************************
// Train the network
//***************************************************************************** 

void dbn::Train(MTRand & random, const MatrixXd& data_E, 
                                 const MatrixXd& data_S) 
{

    MatrixXd newDataset;
    MatrixXd oldDataset = data_E;
    
    for (int l=1; l<layers; ++l) {
        
        cout << "Training RBM " << l << ".." << endl;
        rbms[l-1].train(random,oldDataset);

        newDataset.setZero(data_E.rows(),rbms[l-1].n_h);
        newDataset = propagateDataset(l,oldDataset);
        oldDataset.setZero(data_E.rows(),rbms[l-1].n_h);
        oldDataset = newDataset;
    }
    
    cout << "Training CRBM " << endl << endl;
    crbms[0].train(random,oldDataset,data_S);
}


//*****************************************************************************
// Equilibrate the Top CRBM
//*****************************************************************************

MatrixXd dbn::equilibrateTop(MTRand& random, int steps, MatrixXd& h0) 
{
    
    MatrixXd h_state(batch_size,crbms[0].n_h);
    MatrixXd v_state(batch_size,crbms[0].n_v);
    MatrixXd l_state(batch_size,crbms[0].n_l);
    
    h_state = h0;

    for (int k=0; k<steps; ++k) {

        v_state = crbms[0].sample_visible(random,h_state);
        l_state = crbms[0].sample_label(random,h_state);
        h_state = crbms[0].sample_hidden(random,v_state,l_state);
    }
    
    return h_state;
}


//*****************************************************************************
// Perform Top-Down Inference 
//*****************************************************************************

MatrixXd dbn::topDownInference(MTRand& random, MatrixXd& hTop) 
{
    MatrixXd v_state;
    MatrixXd new_v_state;

    v_state.setZero(batch_size,crbms[0].n_v);

    v_state = crbms[0].sample_visible(random,hTop);

    for (int l=layers-1; l>0; --l) {
        
        new_v_state.setZero(batch_size,rbms[l-1].n_v);
        new_v_state = rbms[l-1].sample_visible(random,v_state);
        v_state.setZero(batch_size,rbms[l-1].n_v);
        v_state = new_v_state;
    }
    
    return v_state;
}


//*****************************************************************************
// Perform Error Correction
//*****************************************************************************

vector<double> dbn::decode(MTRand& random, Decoder& TC,
                           vector<int>& E0, vector<int>& S0){

    batch_size = 1;

    MatrixXd h_top(batch_size,crbms[0].n_h);
    MatrixXd v_state(batch_size,rbms[0].n_v);
    MatrixXd l_state(batch_size,crbms[0].n_l);

    vector<int> E;
    vector<int> C;
    E.assign(rbms[0].n_v,0);
    C.assign(rbms[0].n_v,0);

    //l_state.setZero(batch_size,n_l);

    for(int s=0; s<batch_size; s++) { 
        for (int i=0; i<crbms[0].n_h; i++) {
            h_top(s,i) = random.randInt(1);
        }
    }
    
    for (int k=0; k<crbms[0].n_l; k++) {
        l_state(0,k) = S0[k];
    }

    int n_measure = 100;
    int n_frequency = 2;
    int eq = 2000;

    int corrected = 0;
    int compatible = 0;
    int S_status;
    int C_status;

    h_top = equilibrateTop(random,eq,h_top);

    //E = v_state.cast <int> ();
    do {
        for (int k=0;k<n_measure; k++) {

            h_top = equilibrateTop(random,n_frequency,h_top);
            
            v_state = topDownInference(random,h_top);

            for (int j=0; j<rbms[0].n_v; j++) {

                E[j] = int(v_state(0,j));
            }
            
            S_status = TC.syndromeCheck(E0,E);

            if (S_status == 0) {
                compatible++;

                C = TC.getCycle(E0,E);
                C_status = TC.getLogicalState(C);
                
                if (C_status == 0) {
                    corrected++;
                }
            }

        } 
    } while (compatible == 0);
    
    //cout << "Syndrome Accuracy: ";
    //cout << 100.0*compatible/(1.0*n_measure) << "%\t";
    //cout << "Correction Accuracy: ";
    //cout <<  100.0*corrected/(1.0*compatible) << "%" << endl; 
    vector<double> accuracy;
    accuracy.push_back(100.0*compatible/(1.0*n_measure));
    accuracy.push_back(100.0*corrected/(1.0*compatible));

    return accuracy;
}


//*****************************************************************************
// Load the Network Parameters
//*****************************************************************************

void dbn::loadParameters(string& modelName) {
    
    ifstream file(modelName);

    for (int l=1; l<layers; ++l) {
        
        for (int i=0; i<rbms[l-1].n_h;i++) {
            for (int j=0; j<rbms[l-1].n_v; j++) {
                file >> rbms[l-1].W(i,j);
            }
        }

        for (int j=0; j<rbms[l-1].n_v; j++) {
            file >> rbms[l-1].b(j);
        }
        
        for (int i=0; i<rbms[l-1].n_h; i++) {
            file >> rbms[l-1].c(i);
        }

    }

    for (int i=0; i<crbms[0].n_h;i++) {
        for (int j=0; j<crbms[0].n_v; j++) {
            file >> crbms[0].W(i,j);
        }
    }

    for (int i=0; i<crbms[0].n_h;i++) {
        for (int k=0; k<crbms[0].n_l; k++) {
            file >> crbms[0].U(i,k);
        }
    }

    for (int j=0; j<crbms[0].n_v; j++) {
        file >> crbms[0].b(j);
    }
    
    for (int i=0; i<crbms[0].n_h; i++) {
        file >> crbms[0].c(i);
    }
    
    for (int k=0; k<crbms[0].n_l; k++) {
        file >> crbms[0].d(k);
    }   


}


//*****************************************************************************
// Save the Network Parameters
//*****************************************************************************

void dbn::saveParameters(string& modelName) {
    
    ofstream file(modelName);

    for (int l=1; l<layers; ++l) {
        
        for (int i=0; i<rbms[l-1].n_h;i++) {
            for (int j=0; j<rbms[l-1].n_v; j++) {
                file << rbms[l-1].W(i,j) << " ";
            }
            file << endl;
        }

        file << endl;

        for (int j=0; j<rbms[l-1].n_v; j++) {
            file << rbms[l-1].b(j) << " ";
        }
        
        file << endl << endl;

        for (int i=0; i<rbms[l-1].n_h; i++) {
            file << rbms[l-1].c(i) << " ";
        }

        file << endl << endl;

    }

    file << endl;

    for (int i=0; i<crbms[0].n_h;i++) {
        for (int j=0; j<crbms[0].n_v; j++) {
            file << crbms[0].W(i,j) << " ";
        }
        file << endl;
    }

    file << endl;

    for (int i=0; i<crbms[0].n_h;i++) {
        for (int k=0; k<crbms[0].n_l; k++) {
            file << crbms[0].U(i,k) << " ";
        }
        file << endl;
    }

    file << endl;

    for (int j=0; j<crbms[0].n_v; j++) {
        file << crbms[0].b(j) << " ";
    }
    
    file << endl << endl;

    for (int i=0; i<crbms[0].n_h; i++) {
        file << crbms[0].c(i) << " ";
    }
    
    file << endl << endl;

    for (int k=0; k<crbms[0].n_l; k++) {
        file << crbms[0].d(k) << " ";
    }   


}
