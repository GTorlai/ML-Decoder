#include "rbm.h"

//***********************************************************************
// Constructor 
//***********************************************************************

rbm::rbm(MTRand& random, map<string,float>& parameters,int nV, int nH) 
{
    
    epochs        = int(parameters["ep"]);
    batch_size    = int(parameters["bs"]);
    learning_rate = parameters["lr"];
    L2_par        = parameters["L2"];
    CD_order      = int(parameters["CD"]);
    n_v = nV;
    n_h = nH;

    W.setZero(n_h,n_v);
    b.setZero(n_v);
    c.setZero(n_h);
    
    dW.setZero(n_h,n_v);
    dB.setZero(n_v);
    dC.setZero(n_h);

    double bound = 4.0 * sqrt(6.0/(n_h + n_v));
    double r;

    for (int i=0; i<n_h; i++) {
        
        for (int j=0; j<n_v;j++) {

            r = random.rand(bound);
            W(i,j) = 2.0 * r - bound;
        }
    }
    printNetwork();
}



//***********************************************************************
// Sampling the Hidden Layer 
//***********************************************************************

MatrixXd rbm::hidden_activation(MatrixXd v_state) {
    
    MatrixXd c_batch(batch_size,n_h);

    for(int s=0; s<batch_size;s++) {
        c_batch.row(s) = c;
    }
    
    MatrixXd pre_activation(batch_size,n_h);
    MatrixXd activation(batch_size,n_h);

    pre_activation = v_state * W.transpose() + c_batch;
    activation = sigmoid(pre_activation);
    
    return activation;
}


//***********************************************************************
// Sampling the Visible Layer 
//***********************************************************************

MatrixXd rbm::visible_activation(MatrixXd h_state) {
    
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
// Sampling the Hidden Layer 
//***********************************************************************

MatrixXd rbm::sample_hidden(MTRand & random, MatrixXd v_state) {
    
    MatrixXd activation(batch_size,n_h);
    MatrixXd h_state(batch_size,n_h);
    h_state.setZero();

    activation = hidden_activation(v_state);
    
    h_state = MC_sampling(random,activation);
    
    return h_state;
}


//***********************************************************************
// Sample the Visible Layer
//***********************************************************************

MatrixXd rbm::sample_visible(MTRand & random, MatrixXd h_state) {
    
    MatrixXd activation(batch_size,n_v);
    MatrixXd v_state(batch_size,n_v);

    activation = visible_activation(h_state);
    
    v_state = MC_sampling(random,activation);
    
    return v_state; 
}


//***********************************************************************
// Perform one step of Contrastive Divergence
//***********************************************************************

void rbm::CD_k(MTRand & random, MatrixXd batch) {
    
    double rec_err;

    MatrixXd h_activation(batch_size,n_h);

    MatrixXd h_state(batch_size,n_h);
    MatrixXd v_state(batch_size,n_v);
    
    VectorXd h(n_h);
    VectorXd v(n_v);

    dW.setZero(n_h,n_v);
    dB.setZero(n_v);
    dC.setZero(n_h);

    h_state = sample_hidden(random,batch);
    h_activation = hidden_activation(batch);
    for (int s=0; s<batch_size; s++) {
        //h = h_state.row(s);
        h = h_activation.row(s);
        v = batch.row(s);
        dW += h*v.transpose();
        dB += v;
        dC += h;
    } 
    
    for (int k=0; k<CD_order; k++) {

        v_state = sample_visible(random,h_state);
        h_state = sample_hidden(random,v_state);
    }

    h_activation = hidden_activation(v_state);
     
    for (int s=0; s<batch_size; s++) {
        //h = h_state.row(s);
        h = h_activation.row(s);
        v = v_state.row(s);
        dW += - h*v.transpose();
        dB += -v;
        dC += -h;
    } 
    
    //rec_err = reconstruction_error(batch,h_state);

    //cout << "Reconstruction Error: " << rec_err << endl;

    W += + (learning_rate/batch_size) * dW;
    b += + (learning_rate/batch_size) * dB;
    c += + (learning_rate/batch_size) * dC;
    
}


//***********************************************************************
// Train the Boltzmann Machine
//***********************************************************************

void rbm::train(MTRand & random, MatrixXd dataset) {

    int n_batches = dataset.rows() / batch_size;
    MatrixXd batch(batch_size,n_v);
    
    for (int e=0; e<epochs; e++) {
        cout << "Epoch: " << e << endl;
        for (int b=0; b< n_batches; b++) {
            batch = dataset.block(b*batch_size,0,batch_size,n_v);
            CD_k(random,batch);
        } 
    }
}


//***********************************************************************
// Sample from the Boltzmann Machine
//***********************************************************************

void rbm::sample(MTRand & random,ofstream & output) {
    
    MatrixXd h_state(batch_size,n_h);
    MatrixXd v_state(batch_size,n_v);

    for(int s=0; s<batch_size; s++) { 
        for (int j=0; j<n_v; j++) {
            v_state(s,j) = random.randInt(1);
        }
        for (int i=0; i<n_h; i++) {
            h_state(s,i) = random.randInt(1);
        }
 
    }

    int n_measure = 100000;
    int n_frequency = 2;
    int eq = 50000;

    for (int k=0;k<eq; k++) {

        v_state = sample_visible(random,h_state);
        h_state = sample_hidden(random,v_state);
        
    }

    for (int k=0;k<n_measure; k++) {

        for(int i=0; i<n_frequency; i++) {

            v_state = sample_visible(random,h_state);
            h_state = sample_hidden(random,v_state);
        }
        
        write(output,v_state);
        //printM_file(v_state,output);
    }
}

//*****************************************************************************
// Save the Network Parameters
//*****************************************************************************

void rbm::loadParameters(string& modelName) 
{
        
    ifstream file(modelName);

    for (int i=0; i<n_h;i++) {
        for (int j=0; j<n_v; j++) {
            file >> W(i,j);
        }
    }
    
    for (int j=0; j<n_v; j++) {
        file >> b(j);
    }
    
    for (int i=0; i<n_h; i++) {
        file >> c(i);
    }
   
}

//*****************************************************************************
// Save the Network Parameters
//*****************************************************************************

void rbm::saveParameters(string& modelName) 
{

    ofstream file(modelName);

    for (int i=0; i<n_h;i++) {
        for (int j=0; j<n_v; j++) {
            file << W(i,j) << " ";
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
    
    file.close(); 

}



//***********************************************************************
// Reconstruction Error 
//***********************************************************************

double rbm::reconstruction_error(MatrixXd data, MatrixXd h_state) {

    double err = 0.0;
    double delta;
    
    MatrixXd activation(batch_size,n_h);

    activation = visible_activation(h_state);
    
    for (int s=0; s<batch_size; s++) {
        
        delta = 0.0;

        for (int j=0; j<n_v; j++) {
            delta += (data(s,j)-activation(s,j))*(data(s,j)-activation(s,j)); 
        }

        err += delta;
    }

    err /= batch_size;
    
    return err;
}

//***********************************************************************
// Block Gibbs Sampler
//***********************************************************************

MatrixXd rbm::MC_sampling(MTRand & random, MatrixXd activation) {

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

MatrixXd rbm::sigmoid(MatrixXd matrix) {

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

void rbm::printNetwork() 
{

    cout << "\n\n******************************\n\n" << endl;
    cout << "RESTRICTED BOLTZMANN MACHINE\n\n";
    cout << "Machine Parameter\n\n";
    cout << "\tNumber of Visible Units: " << n_v << "\n";
    cout << "\tNumber of Hidden Units: " << n_h << "\n";
    cout << "\nHyper-parameters\n\n";
    cout << "\tLearning Rate: " << learning_rate << "\n";
    cout << "\tEpochs: " << epochs << "\n";
    cout << "\tBatch Size: " << batch_size << "\n";
    cout << "\tL2 Regularization: " << L2_par << "\n";

    
 
}
