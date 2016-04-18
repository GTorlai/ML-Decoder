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
    CD_order      = int(parameters["CD"]);
    n_v = nV;
    n_h = nH;
    n_l = nL;

//crbm::crbm(MTRand & random) {
//      
//    batch_size=100;
//    learning_rate  = 0.01;
//    CD_order = 15;
//    L2_par = 0.001;
//    epochs = 1; 
//    n_h = 64;
//    n_v = 32;
//    n_l = 16;

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
    
    printNetwork();
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
// Perform one step of Contrastive Divergence
//*****************************************************************************

void crbm::CD_k(MTRand & random, const MatrixXd & batch_V, 
                                 const MatrixXd & batch_L) 
{
    
    double rec_err;

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

    h_state = sample_hidden(random,batch_V,batch_L);
    h_activation = hidden_activation(batch_V,batch_L);
    
    for (int s=0; s<batch_size; s++) {
        //h = h_state.row(s);
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
        //h = h_state.row(s);
        h = h_activation.row(s);
        v = v_state.row(s);
        l = l_state.row(s);
        dW += - h*v.transpose();
        dU += - h*l.transpose();
        dB += -v;
        dC += -h;
        dD += -l;
    } 
    
    W += + (learning_rate/batch_size) * (dW + L2_par*W);
    U += + (learning_rate/batch_size) * (dU + L2_par*U);
    b += + (learning_rate/batch_size) * dB;
    c += + (learning_rate/batch_size) * dC;
    d += + (learning_rate/batch_size) * dD;
    
}

//void crbm::CD_k(MTRand & random, const MatrixXd & batch_V, 
//                                 const MatrixXd & batch_L) 
//{
//    
//    double rec_err;
//    
//    MatrixXd d_w;
//    VectorXd d_b;
//    VectorXd d_c;
//    MatrixXd d_u;
//    VectorXd d_d;
//    d_w.setZero(n_h,n_v);
//    d_u.setZero(n_h,n_l);
//    d_b.setZero(n_v);
//    d_c.setZero(n_h);
//    d_d.setZero(n_l);
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
//    //dW.setZero(n_h,n_v);
//    //dU.setZero(n_h,n_l);
//    //dB.setZero(n_v);
//    //dC.setZero(n_h);
//    //dD.setZero(n_l);
//
//    h_state = sample_hidden(random,batch_V,batch_L);
//    h_activation = hidden_activation(batch_V,batch_L);
//    int s;
//    int chunk=5;
//
//    #pragma omp declare \
//    reduction(MatrixSum : MatrixXd : MatrixXd::operator+))//\: MatrixBase<Derived>)
//
//    #pragma omp parallel for private(s,h,v,l) \
//    schedule(static,chunk) reduction(+:d_w)//,d_u,d_b,d_c,d_d)
//    for (int s=0; s<batch_size; s++) {
//        //h = h_state.row(s);
//        h = h_activation.row(s);
//        v = batch_V.row(s);
//        l = batch_L.row(s);
//        d_w += h*v.transpose();
//        //d_u += h*l.transpose();
//        //d_b += v;
//        //d_c += h;
//        //d_d += l;
//    } 
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
//    //#pragma omp parallel for private(h,v,l) reduction(+:dW,dU,dB,dC,dD) 
//    for (int s=0; s<batch_size; s++) {
//        //h = h_state.row(s);
//        h = h_activation.row(s);
//        v = v_state.row(s);
//        l = l_state.row(s);
//        d_w += - h*v.transpose();
//        d_u += - h*l.transpose();
//        d_b += -v;
//        d_c += -h;
//        d_d += -l;
//    } 
//    
//    W += + (learning_rate/batch_size) * (d_w + L2_par*W);
//    U += + (learning_rate/batch_size) * (d_u + L2_par*U);
//    b += + (learning_rate/batch_size) * d_b;
//    c += + (learning_rate/batch_size) * d_c;
//    d += + (learning_rate/batch_size) * d_d;
//    
//}


//*****************************************************************************
// Train the Boltzmann Machine
//*****************************************************************************

void crbm::train(MTRand & random, const MatrixXd & dataset_V, 
                                  const MatrixXd & dataset_L) 
{

    int n_batches = dataset_V.rows() / batch_size;
    MatrixXd batch_V(batch_size,n_v);
    MatrixXd batch_L(batch_size,n_l);

    for (int e=0; e<epochs; e++) {
        cout << "Epoch: " << e << endl;
        for (int b=0; b< n_batches; b++) {
            batch_V = dataset_V.block(b*batch_size,0,batch_size,n_v);
            batch_L = dataset_L.block(b*batch_size,0,batch_size,n_l);
            CD_k(random,batch_V,batch_L);
        }
        cout << endl << W(0,0) << endl;
    }
}


//*****************************************************************************
// Perform Error Correction
//*****************************************************************************

vector<double> crbm::decode(MTRand & random, Decoder & TC, 
                            vector<int> E0, vector<int> S0) 
{
    
    batch_size = 1;

    MatrixXd h_state(batch_size,n_h);
    MatrixXd v_state(batch_size,n_v);
    MatrixXd l_state(batch_size,n_l);

    vector<int> E;
    vector<int> C;
    E.assign(n_v,0);
    C.assign(n_v,0);

    l_state.setZero(batch_size,n_l);

    for(int s=0; s<batch_size; s++) { 
        for (int j=0; j<n_v; j++) {
            v_state(s,j) = random.randInt(1);
        }
        for (int i=0; i<n_h; i++) {
            h_state(s,i) = random.randInt(1);
        }
    }
    
    for (int k=0; k<n_l; k++) {
        l_state(0,k) = S0[k];
    }

    int n_measure = 100;
    int n_frequency = 2;
    int eq = 2000;
    
    int corrected = 0;
    int compatible = 0;
    int S_status;
    int C_status;

    for (int k=0;k<eq; k++) {
        
        h_state = sample_hidden(random,v_state,l_state);  
        v_state = sample_visible(random,h_state);
    }

    //E = v_state.cast <int> ();
    do {
        for (int k=0;k<n_measure; k++) {

            for(int i=0; i<n_frequency; i++) {
            
                h_state = sample_hidden(random,v_state,l_state);
                v_state = sample_visible(random,h_state);
            }
            
            for (int j=0; j<n_v; j++) {

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

void crbm::printNetwork() 
{

    cout << "\n\n******************************\n\n" << endl;
    cout << "CONDITIONAL RESTRICTED BOLTZMANN MACHINE\n\n";
    cout << "Machine Parameter\n\n";
    cout << "\tNumber of Visible Units: " << n_v << "\n";
    cout << "\tNumber of Hidden Units: " << n_h << "\n";
    cout << "\tNumber of Label Units: " << n_l << "\n";
    cout << "\nHyper-parameters\n\n";
    cout << "\tLearning Rate: " << learning_rate << "\n";
    cout << "\tEpochs: " << epochs << "\n";
    cout << "\tBatch Size: " << batch_size << "\n";
    cout << "\tL2 Regularization: " << L2_par << "\n";

    
 
}






