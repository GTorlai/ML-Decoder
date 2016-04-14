#ifndef UTILITIES_H
#define UTITLITES_H

#include <Eigen/Core>

template<typename T> ostream& operator<< (ostream& out, const MatrixBase<T>& M)
{    
    for (size_t i =0; i< M.rows(); ++i) {
        
        for (size_t j =0; j< M.cols(); ++j) {
            
            out << M(i,j)<< " ";
        }
        
        if (M.cols() > 1) out << endl;
    }
    
    out << endl;

    return out;
}


template<typename T> void write (ofstream& fout,const MatrixBase<T>& M)
{
    for (size_t i =0; i< M.rows(); ++i) {
        
        for (size_t j =0; j< M.cols(); ++j) {
            
            fout << M(i,j)<< " ";
        }
        
        if (M.cols() > 1) fout << endl;
    }
    
    fout << endl;
}

template<typename T> Eigen::MatrixXd sigmoid(const ArrayBase<T>& M)
{
    return M.exp()/(1.0+M.exp());

}


#endif
