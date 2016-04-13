#include "crbm.h"

//***********************************************************************
// Constructor 
//***********************************************************************

crbm::crbm(MTRand & random) {

    epochs = 1;
    batch_size=1000;
    learning_rate  = 0.01;
    CD_order = 1;
    L2_par = 0.0;
    
    n_h = 4;
    n_v = 16;
    n_l = 8;

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


//***********************************************************************
// Hidden Layer Activation 
//***********************************************************************

MatrixXd crbm::hidden_activation(MatrixXd & v_state,MatrixXd & l_state) {
    
    MatrixXd c_batch(batch_size,n_h);

    for(int s=0; s<batch_size;s++) {
        c_batch.row(s) = c;
    }
    
    MatrixXd pre_activation(batch_size,n_h);
    MatrixXd activation(batch_size,n_h);

    pre_activation = v_state * W.transpose() * l_state * U.transpose() + c_batch;
    activation = sigmoid(pre_activation);
    
    return activation;
}


//***********************************************************************
// Visible Layer Activation 
//***********************************************************************

MatrixXd crbm::visible_activation(MatrixXd & h_state) {
    
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

//***********************************************************************
// Label Layer Activation 
//***********************************************************************

MatrixXd crbm::label_activation(MatrixXd & h_state) {
    
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


//***********************************************************************
// Sampling the Hidden Layer 
//***********************************************************************

MatrixXd crbm::sample_hidden(MTRand & random, MatrixXd & v_state, MatrixXd & l_state) {
    
    MatrixXd activation(batch_size,n_h);
    MatrixXd h_state(batch_size,n_h);

    activation = hidden_activation(v_state,l_state);
    
    h_state = MC_sampling(random,activation);
    
    return h_state;
}


//***********************************************************************
// Sample the Visible Layer
//***********************************************************************

MatrixXd crbm::sample_visible(MTRand & random, MatrixXd & h_state) {
    
    MatrixXd activation(batch_size,n_v);
    MatrixXd v_state(batch_size,n_v);

    activation = visible_activation(h_state);
    
    v_state = MC_sampling(random,activation);
    
    return v_state; 
}


//***********************************************************************
// Sample the Label Layer
//***********************************************************************

MatrixXd crbm::sample_label(MTRand & random, MatrixXd & h_state) {
    
    MatrixXd activation(batch_size,n_l);
    MatrixXd l_state(batch_size,n_l);

    activation = label_activation(h_state);
    
    l_state = MC_sampling(random,activation);
    
    return l_state; 
}


//***********************************************************************
// Predict Label 
//***********************************************************************

void crbm::accuracy(MTRand & random, MatrixXd & batch_V,MatrixXd & batch_L) {
    
    MatrixXd h_state(batch_size,n_h);
    MatrixXd v_state(batch_size,n_v);
    MatrixXd l_state;
    
    l_state.setZero(batch_size,n_l);
    
    h_state = sample_hidden(random,batch_V,l_state);
    l_state = sample_label(random,h_state);
 
    int eq = 100;
    int steps = 100;

    for (int n=0; n<eq; n++) {    
        h_state = sample_hidden(random,batch_V,l_state);
        l_state = sample_label(random,h_state);
    }
    
    for (int n=0; n<steps; n++) {    
        h_state = sample_hidden(random,batch_V,l_state);
        l_state = sample_label(random,h_state);
    }
}


//***********************************************************************
// Perform one step of Contrastive Divergence
//***********************************************************************

void crbm::CD_k(MTRand & random, MatrixXd & batch_V, MatrixXd & batch_L) {
    
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
    
    //rec_err = reconstruction_error(batch,h_state);

    //cout << "Reconstruction Error: " << rec_err << endl;

    W += + (learning_rate/batch_size) * dW;
    U += + (learning_rate/batch_size) * dU;
    b += + (learning_rate/batch_size) * dB;
    c += + (learning_rate/batch_size) * dC;
    d += + (learning_rate/batch_size) * dD;
    
}


//***********************************************************************
// Train the Boltzmann Machine
//***********************************************************************

void crbm::train(MTRand & random, MatrixXd & dataset_V, MatrixXd & dataset_L) {

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
    }
}


////***********************************************************************
//// Sample from the Boltzmann Machine
////***********************************************************************
//
//void rbm::sample(MTRand & random,ofstream & output) {
//    
//    MatrixXd h_state(batch_size,n_h);
//    MatrixXd v_state(batch_size,n_v);
//
//    for(int s=0; s<batch_size; s++) { 
//        for (int j=0; j<n_v; j++) {
//            v_state(s,j) = random.randInt(1);
//        }
//        for (int i=0; i<n_h; i++) {
//            h_state(s,i) = random.randInt(1);
//        }
// 
//    }
//
//    int n_measure = 100000;
//    int n_frequency = 2;
//    int eq = 50000;
//
//    for (int k=0;k<eq; k++) {
//
//        v_state = sample_visible(random,h_state);
//        h_state = sample_hidden(random,v_state);
//        
//    }
//
//    for (int k=0;k<n_measure; k++) {
//
//        for(int i=0; i<n_frequency; i++) {
//
//            v_state = sample_visible(random,h_state);
//            h_state = sample_hidden(random,v_state);
//        }
//        
//        printM_file(v_state,output);
//    }
//}


//***********************************************************************
// Block Gibbs Sampler
//***********************************************************************

MatrixXd crbm::MC_sampling(MTRand & random, MatrixXd & activation) {

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


//***********************************************************************
// Sigmoid function
//***********************************************************************

MatrixXd crbm::sigmoid(MatrixXd & matrix) {

    MatrixXd X(matrix.rows(),matrix.cols());
    
    for (int i=0; i< X.rows(); i++) {
        for(int j=0; j<X.cols();j++) {
            X(i,j) = 1.0/(1.0+exp(-matrix(i,j)));
        }
    }

    return X;

}


//***********************************************************************
// Print Matrix
//***********************************************************************

void crbm::printM(MatrixXd matrix) {
    
    for(int i=0; i< matrix.rows(); i++) {
        for(int j=0;j<matrix.cols(); j++) {

            cout << matrix(i,j) << " ";
        }
        cout << endl;
    }

}


//***********************************************************************
// Print Matrix on File 
//***********************************************************************

void crbm::printM_file(MatrixXd matrix, ofstream & file) {
    
    for(int i=0; i< matrix.rows(); i++) {
        for(int j=0;j<matrix.cols(); j++) {

            file << matrix(i,j) << " ";
        }
        file << endl;
    }

}

