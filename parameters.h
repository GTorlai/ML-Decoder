#ifndef PARAMETERS_H
#define PARAMETERS_H

class Parameters {
    public:
    
    int n_v;
    int n_h;
    int n_l;

    int epochs;
    int batch_size;
    int CD_order;

    float learning_rate;
    float L2Reg;

    Parameters(v,h,l,e,bs,cd,lr,L2);             

};

//Constructor
Parameters::Parameters(v,h,l,e,bs,cd,lr,L2) {

    n_v = v;
    n_h = h;
    n_l = l;

    epochs = e;
    batch_size = bs;
    CD_order = cd;

    learning_rate = lr;
    L2Reg = L2;
}

#endif
