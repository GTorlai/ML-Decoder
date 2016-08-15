#include "ToricCode.h"

Decoder::Decoder(int L_) {

    L = L_;
    N = L*L;
    Nqubits = 2*N;

    int counter[2]={0,0};
    
    Coordinates.resize(N,vector<int>(2));
    Neighbors.resize(N,vector<int>(4));
    plaqQubits.resize(N,vector<int>(4));
    starQubits.resize(N,vector<int>(4));

    for (int i=1;i<N;i++){
        
        if ( (counter[0]+1) % L == 0){ //end of x-row
            
            //coordinate[2*i] = 0; //reset
            counter[0]=0;
            
                if ( (counter[1]+1) % L == 0)
                    counter[1] = 0; //reset
                else{
                    counter[1]++;
                    //break;
                }
        }//if
        else {
            counter[0]++;
        }
        
        Coordinates[i][0]=counter[0];
        Coordinates[i][1]=counter[1];
    }
    
    //Neighbours
    for(int i=0;i<N;i++) {
        
        Neighbors[i][0]=index(Coordinates[i][0]+1,Coordinates[i][1]);
        Neighbors[i][1]=index(Coordinates[i][0]  ,Coordinates[i][1]+1);
        Neighbors[i][2]=index(Coordinates[i][0]-1,Coordinates[i][1]);
        Neighbors[i][3]=index(Coordinates[i][0]  ,Coordinates[i][1]-1);
    }
    
    //4 links on a plaquette
    
    for(int x=0;x<L;x++) {
        for(int y=0;y<L;y++) {
                
            plaqQubits[index(x,y)][0]=2*index(x,y);
            plaqQubits[index(x,y)][3]=2*index(x,y)+1;
            plaqQubits[index(x,y)][1]=2*index(x+1,y)+1;
            plaqQubits[index(x,y)][2]=2*index(x,y+1);
        }
    }
    
    //4 Links on a site
    for(int i=0;i<N;i++) {
        
        starQubits[i][0]=2*i;
        starQubits[i][1]=2*i+1;
        starQubits[i][2]=2*Neighbors[i][2];
        starQubits[i][3]=2*Neighbors[i][3]+1;
    }


}


vector<int> Decoder::generateError(MTRand & random, double p) {

    vector<int> E;
    E.assign(Nqubits,0);

    for (int i=0; i<Nqubits; i++) {

        if (random.rand() < p) E[i] = 1;
    }

    return E;
}


int Decoder::measureStar(vector<int> E, int star) {

    int stabilizer = 1;

    for (int i=0; i<4; i++) {

        stabilizer *= -(2*E[starQubits[star][i]]-1);
    }

    return stabilizer;
}


vector<int> Decoder::getSyndrome(vector<int> E) {

    vector<int> S;
    S.assign(N,1);

    int stabilizer;
    
    for (int i=0; i<N; i++) {
        stabilizer = measureStar(E,i);
        if (stabilizer == 1) S[i] = 0;
    }

    return S;
}

vector<int> Decoder::getCycle(vector<int> E0, vector<int> E) {

    vector<int> C;
    C.assign(Nqubits,1);

    for (int i=0; i<Nqubits; i++) {

        if (E0[i] == E[i]) C[i] = 0;
    }

    return C;
}


int Decoder::syndromeCheck(vector<int> E0, vector<int> E) {

    int status = 0;

    vector<int> S0 = getSyndrome(E0);
    vector<int> S = getSyndrome(E);

    for (int i=0; i< N; i++) {

        if (S0[i] != S[i]) {
            status = 1;
            break;
        }
    }

    return status;
}


int Decoder::getLogicalState(vector<int> C) {

    int status = 0;

    for (int x=0; x<L; x++) {

        int temp = 0;

        for (int y=0; y<L; y++) {

            temp += C[starQubits[index(x,y)][0]];
        }

        if ((temp % 2) != 0) {
            status = 1;
            break;
        }
    }

    return status;
}


void Decoder::testDecoder(MTRand & random) {

    vector<int> E;
    vector<int> S;
    vector<int> E0;
    vector<int> S0;
    vector<int> C;
    

    int right_recovery = 1000;
    int wrong_recovery = 888; 
    
    int right_counter=0;
    int wrong_counter=0;


    E0 = generateError(random,0.1);
    S0 = getSyndrome(E0);
    
    E.assign(Nqubits,0);

    for (int i=0; i<Nqubits; i++) {
        E[i] = E0[i];
    }

    int plaq;
    int loop;
    int S_status;
    int C_status;
    for (int k=0; k<right_recovery; k++) {

        plaq = random.randInt(N-1);

        for (int i=0; i<4; i++) {
            E[plaqQubits[plaq][i]] ^=1;
        }
        
        C = getCycle(E0,E);
        C_status = getLogicalState(C);
        if (C_status ==0) right_counter++;
    }
    
    for (int k=0; k<wrong_recovery; k++) {

        plaq = random.randInt(N-1);
        loop = random.randInt(L-1);

        for (int i=0; i<L; i++) {
            E[2*(L*loop + i)] ^= 1;
        }
        
        C = getCycle(E0,E);
        C_status = getLogicalState(C);
        
        if (C_status ==1) wrong_counter++;
 
        for (int i=0; i<L; i++) {
            E[2*(L*loop + i)] ^= 1;
        }

        for (int i=0; i<4; i++) {
            E[plaqQubits[plaq][i]] ^=1;
        }
        
    }

    cout << "Number of right states: " << right_counter << endl;
    cout << "Number of wrong states: " << wrong_counter << endl;
 
    //S_status = syndromeCheck(E0,E);
    //
    //cout << "Syndrome ";
    //if (S_status == 0) cout << " CORRECT" << endl << endl;
    //else cout << " INCORRECT" << endl << endl;

    
    
    //cout << "Logical State: ";
    //if (C_status == 0) cout << " PROTECTED" << endl << endl;
    //else cout << "CORRUPTED" << endl << endl;
    

}


//Indexing of coordinates
int Decoder::index(int x, int y) {

    if (x<0) x+= L;
    if (x>=L) x-= L;
    if (y<0) y+= L;
    if (y>=L) y-= L;

    return L*y+x;

}

//Print Lattice Informations
void Decoder::printToricCodeInfo() {

    cout << endl << endl << "Printing Indexing of Coordinates..." << endl << endl;

    for (int x=0; x<L; x++) {
        for (int y=0; y<L; y++) {
            cout << "Coordinate (" << x << "," << y << ")";
            cout << "  -->  Index: " << index(x,y) << endl;
        }
    }

    cout << endl << endl << endl;

    cout << "Printing coordinates of indices..." << endl << endl;
    
    for (int i=0; i<N; i++) {

        cout << "Index "<<i;
        cout << "  --> Coordinates: (";
        cout << Coordinates[i][0] <<","<<Coordinates[i][1]<<")";
        cout << endl;
    }
    cout << endl << endl << endl;

    cout << "Printing neighbors..." << endl << endl;
    
    for (int i=0; i<N; i++) {

        cout << "Index "<<i;
        cout << "  --> Neighbors: ";
        
        for (int j=0; j<4;j++) {
            cout << Neighbors[i][j] << " , ";
        }
        
        cout << endl;
    }
    cout << endl << endl << endl;
    
    cout << "Printing links on plaquettes..." << endl << endl;
 
    for (int i=0; i<N; i++) {

        cout << "Plaquette "<<i;
        cout << "  --> Links: ";
        
        for (int j=0; j<4;j++) {
            cout << plaqQubits[i][j] << " , ";
        }
        
        cout << endl;
    }
    cout << endl << endl << endl;
    
    cout << "Printing links on stars..." << endl << endl;
 
    for (int i=0; i<N; i++) {

        cout << "Star "<<i;
        cout << "  --> Links: ";
        
        for (int j=0; j<4;j++) {
            cout << starQubits[i][j] << " , ";
        }
        
        cout << endl;
    }
    cout << endl << endl << endl;

}


//Print Vector on terminal
void Decoder::printVector(vector<int> Vector){

    for (int i=0;i<Vector.size();i++){
        cout<<Vector[i]<<" ";
    }//i
    cout<<endl;

}//print

//Write Vector on file
void Decoder::writeVector(vector<int> Vector, ofstream & file){

   for (int i=0;i<Vector.size();i++){
        file<<Vector[i]<<" ";
    }//i
    file<<endl; 
}
