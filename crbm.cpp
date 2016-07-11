#include "crbm.h"
#include <string.h>
#include <omp.h>

//*****************************************************************************
// Constructor 
//*****************************************************************************

crbm::crbm(MTRand & random, map<string,float>& parameters,
           int nV, int nH, int nL) 
{
    
    epochs        = int(parameters["ep"]);
    batch_size    = int(parameters["bs"]);
    learning_rate = parameters["lr"];
    L2_par        = parameters["L2"];
    p_drop        = parameters["p_drop"];
    //beta          = parameters["beta"]; 
    CD_order      = int(parameters["CD"]); 
    
    //if (int(parameters["PCD"]) > 0) {
    //    CD_order = int(parameters["PCD"]);
    //    CD_type = "persistent"; 
    //}

    //else {
    //    CD_order = int(parameters["CD"]);
    //    CD_type = "default";
    //}
    //
    if (p_drop > 0) regularization = "Dropout";
    if (L2_par > 0) regularization = "Weight Decay";
    alpha = 0.9;
    
    n_v = nV;
    n_h = nH;
    n_l = nL;
    
    //Persistent.setZero(batch_size,n_h);

    W.setZero(n_h,n_v);
    U.setZero(n_h,n_l);
    b.setZero(n_v);
    c.setZero(n_h);
    d.setZero(n_l);
    
    double bound_v = 4.0 * sqrt(6.0/(n_h + n_v));
    double bound_l = 4.0 * sqrt(6.0/(n_h + n_l));
    double r;

    for (int i=0; i<n_h; i++) {
        for (int j=0; j<n_v;j++) {
            r = random.rand(bound_v);
            W(i,j) = 2.0 * r - bound_v;
        }
        for (int k=0; k<n_l;k++) {
            r = random.rand(bound_l);
            U(i,k) = 2.0 * r - bound_l;
        }
    }
    
}


//*****************************************************************************
// Hidden Layer Activation 
//*****************************************************************************

MatrixXd crbm::hidden_activation(const MatrixXd & v_state,
                                 const MatrixXd & l_state) 
{
    
    MatrixXd c_batch(batch_size,n_h);

    for(int s=0; s<batch_size;s++) {
        c_batch.row(s) = c;
    }
    
    MatrixXd pre_activation(batch_size,n_h);
    MatrixXd activation(batch_size,n_h);

    pre_activation = v_state * W.transpose() + l_state * U.transpose() + c_batch;
    activation = sigmoid(pre_activation);
    
    return activation;
}


//*****************************************************************************
// Visible Layer Activation 
//*****************************************************************************

MatrixXd crbm::visible_activation(const MatrixXd & h_state) 
{
    
    MatrixXd b_batch(batch_size,n_v);

    for(int s=0; s<batch_size;s++) {
        b_batch.row(s) = b;
    }

    MatrixXd pre_activation(batch_size,n_v);
    MatrixXd activation(batch_size,n_v);
    
    pre_activation = h_state * W + b_batch;
    activation = sigmoid(pre_activation);
    
    return activation; 
}


//*****************************************************************************
// Label Layer Activation 
//*****************************************************************************

MatrixXd crbm::label_activation(const MatrixXd & h_state) 
{
    
    MatrixXd d_batch(batch_size,n_l);

    for(int s=0; s<batch_size;s++) {
        d_batch.row(s) = d;
    }

    MatrixXd pre_activation(batch_size,n_l);
    MatrixXd activation(batch_size,n_l);
    
    pre_activation = h_state * U + d_batch;
    activation = sigmoid(pre_activation);
    
    return activation; 
}


//*****************************************************************************
// Build the dropout asmk for the hidden layer
//*****************************************************************************

MatrixXd crbm::buildDropoutMask(MTRand & random)
{
    MatrixXd mask(batch_size,n_h);
    
    for(int s=0; s<batch_size; ++s) {
        for (int i=0; i<n_h; ++i) {
            if (random.rand() < p_drop) mask(s,i) = 1;
            else mask(s,i) = 0;       
        }
    }
    
    return mask;

}


//*****************************************************************************
// Sampling the Hidden Layer 
//*****************************************************************************

MatrixXd crbm::sample_hidden(MTRand & random, const MatrixXd & v_state, 
                                              const MatrixXd & l_state) 
{
    
    MatrixXd activation(batch_size,n_h);
    MatrixXd h_state(batch_size,n_h);

    activation = hidden_activation(v_state,l_state);
    
    h_state = MC_sampling(random,activation);
    
    return h_state;
}


//*****************************************************************************
// Sample the Visible Layer
//*****************************************************************************

MatrixXd crbm::sample_visible(MTRand & random, const MatrixXd & h_state) {
    
    MatrixXd activation(batch_size,n_v);
    MatrixXd v_state(batch_size,n_v);

    activation = visible_activation(h_state);
    
    v_state = MC_sampling(random,activation);
    
    return v_state; 
}


//*****************************************************************************
// Sample the Label Layer
//*****************************************************************************

MatrixXd crbm::sample_label(MTRand & random, const MatrixXd & h_state) 
{
    
    MatrixXd activation(batch_size,n_l);
    MatrixXd l_state(batch_size,n_l);

    activation = label_activation(h_state);
    
    l_state = MC_sampling(random,activation);
    
    return l_state; 
}

//*****************************************************************************
// Build the dropout asmk for the hidden layer
//*****************************************************************************

MatrixXd crbm::dropout(MTRand & random, const MatrixXd& mask, 
                       const MatrixXd& v_state, const MatrixXd& l_state)
{
    
    MatrixXd activation(batch_size,n_h);
    MatrixXd h_state(batch_size,n_h);
    //ArrayXXd dropH(batch_size,n_h);
    MatrixXd dropH;
    dropH.setZero(batch_size,n_h);

    activation = hidden_activation(v_state,l_state);
    
    h_state = MC_sampling(random,activation);
    
    for (int s=0; s<batch_size; ++s) {
        for (int i=0; i<n_h; ++i) {

            if (mask(s,i) == 0)
               h_state(s,i) = 0; 
        }
    }

    //dropH = (mask.array())*(h_state.array());

    //return dropH.matrix();

    return h_state; 
}



//*****************************************************************************
// Perform one step of Contrastive Divergence
//*****************************************************************************

void crbm::genCD(MTRand & random, const MatrixXd & batch_V, const MatrixXd & batch_L) 
{
    
    MatrixXd h_activation(batch_size,n_h);
    
    MatrixXd h_state(batch_size,n_h);
    MatrixXd v_state(batch_size,n_v);
    MatrixXd l_state(batch_size,n_l);

    VectorXd h(n_h);
    VectorXd v(n_v);
    VectorXd l(n_l);

    dW.setZero(n_h,n_v);
    dU.setZero(n_h,n_l);
    dB.setZero(n_v);
    dC.setZero(n_h);
    dD.setZero(n_l);
    
    //if (CD_type.compare("persistent") == 0) {
    //h_state = Persistent;
    //}
    //else {
    //    h_state = sample_hidden(random,batch_V,batch_L);
    //}
    

    h_state = sample_hidden(random,batch_V,batch_L);
    h_activation = hidden_activation(batch_V,batch_L);
    
    for (int s=0; s<batch_size; s++) {
        h = h_activation.row(s);
        v = batch_V.row(s);
        l = batch_L.row(s);
        dW += h*v.transpose();
        dU += h*l.transpose();
        dB += v;
        dC += h;
        dD += l;
    } 
    
    for (int k=0; k<CD_order; k++) {

        v_state = sample_visible(random,h_state);
        l_state = sample_label(random,h_state);
        h_state = sample_hidden(random,v_state,l_state);
    }

    h_activation = hidden_activation(v_state,l_state);
    
    for (int s=0; s<batch_size; s++) {
        h = h_activation.row(s);
        v = v_state.row(s);
        l = l_state.row(s);
        dW += - h*v.transpose();
        dU += - h*l.transpose();
        dB += -v;
        dC += -h;
        dD += -l;
    } 
    
    W += + (learning_rate/batch_size) * (dW - L2_par*W);
    U += + (learning_rate/batch_size) * (dU - L2_par*U);
    b += + (learning_rate/batch_size) * dB;
    c += + (learning_rate/batch_size) * dC;
    d += + (learning_rate/batch_size) * dD;
    
    //if (CD_type.compare("persistent") == 0) {
    //Persistent = h_state;
    //}
 
}

//*****************************************************************************
// Perform one step of Contrastive Divergence
//*****************************************************************************

void crbm::dropCD(MTRand & random, const MatrixXd & batch_V, const MatrixXd & batch_L) 
{
    
    MatrixXd h_activation(batch_size,n_h);
    MatrixXd mask(batch_size,n_h);
        
    MatrixXd h_state(batch_size,n_h);
    MatrixXd v_state(batch_size,n_v);
    MatrixXd l_state(batch_size,n_l);

    VectorXd h(n_h);
    VectorXd v(n_v);
    VectorXd l(n_l);

    dW.setZero(n_h,n_v);
    dU.setZero(n_h,n_l);
    dB.setZero(n_v);
    dC.setZero(n_h);
    dD.setZero(n_l);
    
    mask = buildDropoutMask(random);

    h_state = dropout(random,mask,batch_V,batch_L);

    h_activation = hidden_activation(batch_V,batch_L);
    
    for (int s=0; s<batch_size; s++) {
        h = h_activation.row(s);
        v = batch_V.row(s);
        l = batch_L.row(s);
        dW += h*v.transpose();
        dU += h*l.transpose();
        dB += v;
        dC += h;
        dD += l;
    } 
    
    for (int k=0; k<CD_order; k++) {

        v_state = sample_visible(random,h_state);
        l_state = sample_label(random,h_state);
        h_state = dropout(random,mask,v_state,l_state);
    }

    h_activation = hidden_activation(v_state,l_state);
    
    for (int s=0; s<batch_size; s++) {
        h = h_activation.row(s);
        v = v_state.row(s);
        l = l_state.row(s);
        dW += - h*v.transpose();
        dU += - h*l.transpose();
        dB += -v;
        dC += -h;
        dD += -l;
    } 
    
    W = 0.95 * W + (learning_rate/batch_size) * dW;
    U = 0.95 * U + (learning_rate/batch_size) * dU;
    b = 0.95 * b + (learning_rate/batch_size) * dB;
    c = 0.95 * c + (learning_rate/batch_size) * dC;
    d = 0.95 * d + (learning_rate/batch_size) * dD;
    
}



//*****************************************************************************
// Train the Boltzmann Machine
//*****************************************************************************

void crbm::train(MTRand & random, const string& network,
        const MatrixXd & dataset_V, const MatrixXd & dataset_L) 
{

    int n_batches = dataset_V.rows() / batch_size;
    MatrixXd batch_V(batch_size,n_v);
    MatrixXd batch_L(batch_size,n_l);
    
    //if (CD_type.compare("persistent") == 0) {
    //    batch_V = dataset_V.block(0,0,batch_size,n_v);
    //    batch_L = dataset_L.block(0,0,batch_size,n_l);
    //    Persistent = sample_hidden(random,batch_V,batch_L); 
    //}
    
    //if (network.compare("CRBM")==0) { 
    if (regularization.compare("Weight Decay") ==0) {
        cout << endl << endl;
        cout << "Training with Weight Decay..." << endl << endl;
        for (int e=0; e<epochs; e++) {
            cout << "Epoch: " << e << endl;
            for (int b=0; b< n_batches; b++) {
                batch_V = dataset_V.block(b*batch_size,0,batch_size,n_v);
                batch_L = dataset_L.block(b*batch_size,0,batch_size,n_l);
                genCD(random,batch_V,batch_L);
            }
        }
    }

    if (regularization.compare("Dropout")==0) {
        cout << endl << endl;
        cout << "Training with Dropout..." << endl << endl;
        for (int e=0; e<epochs; e++) {
            cout << "Epoch: " << e << endl;
            for (int b=0; b< n_batches; b++) {
                batch_V = dataset_V.block(b*batch_size,0,batch_size,n_v);
                batch_L = dataset_L.block(b*batch_size,0,batch_size,n_l);
                dropCD(random,batch_V,batch_L);
            }
        }
    }
    //}
   // 
   // if (network.compare("discCRBM")==0) { 
   //     if (regularization.compare("Weigth Decay") ==0) {
   //         for (int e=0; e<epochs; e++) {
   //             cout << "Epoch: " << e << endl;
   //             for (int b=0; b< n_batches; b++) {
   //                 batch_V = dataset_V.block(b*batch_size,0,batch_size,n_v);
   //                 batch_L = dataset_L.block(b*batch_size,0,batch_size,n_l);
   //                 discCD(random,batch_V,batch_L);
   //             }
   //         }
   //     }

   //     //if (regularization.compare("Dropout")==0) {
   //     //    for (int e=0; e<epochs; e++) {
   //     //        cout << "Epoch: " << e << endl;
   //     //        for (int b=0; b< n_batches; b++) {
   //     //            batch_V = dataset_V.block(b*batch_size,0,batch_size,n_v);
   //     //            batch_L = dataset_L.block(b*batch_size,0,batch_size,n_l);
   //     //            dropCD(random,batch_V,batch_L);
   //     //        }
   //     //    }
   //     //}
   // }

   // if (network.compare("hybridCRBM")==0) { 
   //     if (regularization.compare("Weigth Decay") ==0) {
   //         for (int e=0; e<epochs; e++) {
   //             cout << "Epoch: " << e << endl;
   //             for (int b=0; b< n_batches; b++) {
   //                 batch_V = dataset_V.block(b*batch_size,0,batch_size,n_v);
   //                 batch_L = dataset_L.block(b*batch_size,0,batch_size,n_l);
   //                 hybridCD(random,beta,batch_V,batch_L);
   //             }
   //         }
   //     }

   //     //if (regularization.compare("Dropout")==0) {
   //     //    for (int e=0; e<epochs; e++) {
   //     //        cout << "Epoch: " << e << endl;
   //     //        for (int b=0; b< n_batches; b++) {
   //     //            batch_V = dataset_V.block(b*batch_size,0,batch_size,n_v);
   //     //            batch_L = dataset_L.block(b*batch_size,0,batch_size,n_l);
   //     //            dropCD(random,batch_V,batch_L);
   //     //        }
   //     //    }
   //     //}
   // }
 
}


//*****************************************************************************
// Perform Error Correction
//*****************************************************************************

double crbm::decode(MTRand & random, Decoder & TC, 
                            MatrixXd& testSet_E, MatrixXd& testSet_S) 
{
    //int size = testSet_E.rows();
    int size = 10000;
    batch_size = 1;
    int n_frequency = 10;
    int eq = 500;
    
    int corrected = 0;
    int S_status;
    int C_status;
    int counter=0;

    MatrixXd h_state(batch_size,n_h);
    MatrixXd v_state(batch_size,n_v);
    MatrixXd l_state(batch_size,n_l);
    
    vector<int> E;
    vector<int> E0;
    vector<int> C;
    E.assign(n_v,0);
    E0.assign(n_v,0);
    C.assign(n_v,0);

    for (int s=0; s<size; ++s) {
        
        counter = 0;

       // cout << s; 
        for (int j=0; j<n_v; ++j) {
            
            v_state(0,j) = random.randInt(1);
            E0[j] = int(testSet_E(s,j));
        }
        
        for (int k=0; k<n_l; ++k) {
            
            l_state(0,k) = testSet_S(s,k);
        }

        for (int n=0; n<eq; ++n) {
        
            h_state = sample_hidden(random,v_state,l_state);  
            v_state = sample_visible(random,h_state);
        }
        
         
        do {
            
            for(int i=0; i<n_frequency; ++i) {
            
                h_state = sample_hidden(random,v_state,l_state);
                v_state = sample_visible(random,h_state);
            }
            
            for (int j=0; j<n_v; ++j) {
                    
                E[j] = int(v_state(0,j));
            }
 
            S_status = TC.syndromeCheck(E0,E);
            
            counter++;
            
        } while ((S_status != 0) && (counter < 10000));
        
        if (S_status == 0) {

            C = TC.getCycle(E0,E);
            C_status = TC.getLogicalState(C);

            if (C_status == 0) {
        //        cout << " -> CORRECTED" << endl;
                corrected++;
            }
            
      //      else cout << " -> FAILED" << endl;

        }
        
    }
    
    //cout << "\n\nAccuracy: " << 100.0*corrected/(1.0*size) << endl;    
    double accuracy = 100.0*corrected/(1.0*size);

    return accuracy;
}


//*****************************************************************************
// Load the Network Parameters
//*****************************************************************************

void crbm::loadParameters(string& modelName) 
{
        
    ifstream file(modelName);

    for (int i=0; i<n_h;i++) {
        for (int j=0; j<n_v; j++) {
            file >> W(i,j);
        }
    }

    for (int i=0; i<n_h;i++) {
        for (int k=0; k<n_l; k++) {
            file >> U(i,k);
        }
    }

    for (int j=0; j<n_v; j++) {
        file >> b(j);
    }
    
    for (int i=0; i<n_h; i++) {
        file >> c(i);
    }
    
    for (int k=0; k<n_l; k++) {
        file >> d(k);
    }   
  



}

//*****************************************************************************
// Save the Network Parameters
//*****************************************************************************

void crbm::saveParameters(string& modelName) 
{

    ofstream file(modelName);

    for (int i=0; i<n_h;i++) {
        for (int j=0; j<n_v; j++) {
            file << W(i,j) << " ";
        }
        file << endl;
    }

    file << endl << endl;

    for (int i=0; i<n_h;i++) {
        for (int k=0; k<n_l; k++) {
            file << U(i,k) << " ";
        }
        file << endl;
    }

    file << endl << endl;

    for (int j=0; j<n_v; j++) {
        file << b(j) << " ";
    }
    
    file << endl << endl;

    for (int i=0; i<n_h; i++) {
        file << c(i) << " ";
    }
    
    file << endl << endl;

    for (int k=0; k<n_l; k++) {
        file << d(k) << " ";
    }   
    
    file.close(); 

}


//*****************************************************************************
// Block Gibbs Sampler
//*****************************************************************************

MatrixXd crbm::MC_sampling(MTRand & random, MatrixXd & activation) 
{

    MatrixXd samples;

    samples.setZero(activation.rows(),activation.cols());

    double r;

    for (int b=0; b<activation.rows(); b++) {

        for (int i=0; i<activation.cols(); i++) {
            
            r = random.rand();
            
            if (r < activation(b,i)) {
                
                samples(b,i) = 1;
            }
        }
    }
    
    return samples;
}


//*****************************************************************************
// Sigmoid function
//*****************************************************************************

MatrixXd crbm::sigmoid(MatrixXd & matrix) 
{

    MatrixXd X(matrix.rows(),matrix.cols());
    
    for (int i=0; i< X.rows(); i++) {
        for(int j=0; j<X.cols();j++) {
            X(i,j) = 1.0/(1.0+exp(-matrix(i,j)));
        }
    }

    return X;

}


//*****************************************************************************
// Print Network Informations
//*****************************************************************************

void crbm::printNetwork(const string& network) 
{
    cout << "\n\n******************************\n\n" << endl;
    cout << "CONDITIONAL RESTRICTED BOLTZMANN MACHINE\n\n";
    cout << "Machine Parameter\n\n";
    cout << "\tNumber of Visible Units: " << n_v << "\n";
    cout << "\tNumber of Hidden Units: " << n_h << "\n";
    cout << "\tNumber of Label Units: " << n_l << "\n";
    cout << "\nHyper-parameters\n\n";
    cout << "\tTraining Objective: ";
    if (network.compare("CRBM")==0) {
        cout << "Generative\n";
    }
    else if (network.compare("discCRBM")==0) {
        cout << "Discriminative\n";
    }
    else if (network.compare("hybridCRBM")==0) {
        cout << "Hybrid with mixing rate of ";
        cout << beta << endl;
    }

    cout << "\tLearning Rate: " << learning_rate << "\n";
    cout << "\tEpochs: " << epochs << "\n";
    cout << "\tBatch Size: " << batch_size << "\n";
    if (regularization.compare("Weight Decay") == 0) {
        cout << "\tRegularization: Weight Decay with L2 = "<<L2_par<< "\n";
    } 
    if (regularization.compare("Dropout") == 0) {
        cout << "\tRegularization: Dropout with probability p= "<<p_drop<< "\n";
    } 
    cout << "\tMomentum: " << alpha << "\n"; 
    if (CD_type.compare("persistent") == 0) {
        cout << "\tOptimization: Persistent Constrastive Divergence" << "\n";
        cout << "\tPCD order: " << CD_order << "\n";
    }
    if (CD_type.compare("default") == 0) {
        cout << "\tOptimization: Constrastive Divergence" << "\n";
        cout << "\tCD order: " << CD_order << "\n";
    }
}

//*****************************************************************************
// Perform one step of Contrastive Divergence
//*****************************************************************************

//void crbm::discCD(MTRand & random, const MatrixXd & batch_V, const MatrixXd & batch_L) 
//{
//    
//    MatrixXd h_activation(batch_size,n_h);
//    
//    MatrixXd h_state(batch_size,n_h);
//    MatrixXd v_state(batch_size,n_v);
//    MatrixXd l_state(batch_size,n_l);
//
//    VectorXd h(n_h);
//    VectorXd v(n_v);
//    VectorXd l(n_l);
//
//    dW.setZero(n_h,n_v);
//    dU.setZero(n_h,n_l);
//    dB.setZero(n_v);
//    dC.setZero(n_h);
//    dD.setZero(n_l);
//    
//    h_state = sample_hidden(random,batch_V,batch_L);
//
//    h_activation = hidden_activation(batch_V,batch_L);
//    
//    for (int s=0; s<batch_size; s++) {
//        h = h_activation.row(s);
//        v = batch_V.row(s);
//        l = batch_L.row(s);
//        dW += h*v.transpose();
//        dU += h*l.transpose();
//        dC += h;
//        dD += l;
//    } 
//    
//
//    l_state = sample_label(random,h_state);
//    h_state = sample_hidden(random,batch_V,l_state);
//
//    h_activation = hidden_activation(batch_V,l_state);
//    
//    for (int s=0; s<batch_size; s++) {
//        h = h_activation.row(s);
//        v = batch_V.row(s);
//        l = l_state.row(s);
//        dW += - h*v.transpose();
//        dU += - h*l.transpose();
//        dC += -h;
//        dD += -l;
//    } 
//    
//    W = alpha*W + (learning_rate/batch_size) * (dW - L2_par*W);
//    U = alpha*U + (learning_rate/batch_size) * (dU - L2_par*U);
//    c = alpha*c + (learning_rate/batch_size) * dC;
//    d = alpha*d + (learning_rate/batch_size) * dD;
//    
//}
//
//void crbm::hybridCD(MTRand & random, double beta,
//        const MatrixXd & batch_V, const MatrixXd & batch_L) 
//{
//    
//    MatrixXd h_activation(batch_size,n_h);
//    
//    MatrixXd h_state0(batch_size,n_h);
//    MatrixXd h_state(batch_size,n_h);
//    MatrixXd v_state(batch_size,n_v);
//    MatrixXd l_state(batch_size,n_l);
//
//    VectorXd h(n_h);
//    VectorXd v(n_v);
//    VectorXd l(n_l);
//
//    dW.setZero(n_h,n_v);
//    dU.setZero(n_h,n_l);
//    dB.setZero(n_v);
//    dC.setZero(n_h);
//    dD.setZero(n_l);
//    
//    h_state0 = sample_hidden(random,batch_V,batch_L);
//    h_state = h_state0;
//
//    h_activation = hidden_activation(batch_V,batch_L);
//    
//    for (int s=0; s<batch_size; s++) {
//        h = h_activation.row(s);
//        v = batch_V.row(s);
//        l = batch_L.row(s);
//        dW += (1+beta)*h*v.transpose();
//        dU += (1+beta)*h*l.transpose();
//        dB += beta*v;
//        dC += (1+beta)*h;
//        dD += (1+beta)*l;
//    } 
//    
//
//    //v_state = sample_visible(random,h_state);
//    l_state = sample_label(random,h_state);
//    h_state = sample_hidden(random,batch_V,l_state);
//
//    h_activation = hidden_activation(batch_V,l_state);
//    
//    for (int s=0; s<batch_size; s++) {
//        h = h_activation.row(s);
//        v = batch_V.row(s);
//        l = l_state.row(s);
//        dW += -h*v.transpose();
//        dU += -h*l.transpose();
//        dC += -h;
//        dD += -l;
//    } 
//    
//    h_state = h_state0;
//
//    for (int k=0; k<CD_order; k++) {
//
//        v_state = sample_visible(random,h_state);
//        l_state = sample_label(random,h_state);
//        h_state = sample_hidden(random,v_state,l_state);
//    }
//
//    h_activation = hidden_activation(v_state,l_state);
//    
//    for (int s=0; s<batch_size; s++) {
//        h = h_activation.row(s);
//        v = v_state.row(s);
//        l = l_state.row(s);
//        dW += -beta*h*v.transpose();
//        dU += -beta*h*l.transpose();
//        dB += -beta*v;
//        dC += -beta*h;
//        dD += -beta*l;
//    } 
// 
//    W = alpha*W + (learning_rate/batch_size) * (dW + L2_par*W);
//    U = alpha*U + (learning_rate/batch_size) * (dU + L2_par*U);
//    b = alpha*b + (learning_rate/batch_size) * dB;
//    c = alpha*c + (learning_rate/batch_size) * dC;
//    d = alpha*d + (learning_rate/batch_size) * dD;
//    
//}




