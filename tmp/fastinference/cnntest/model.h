#pragma once 

namespace FAST_INFERENCE {    
    #define REF_ACCURACY 98.49

    #define N_CLASSES 10
    #define N_FEATURES 784

    void predict_cnntest(double const * const x, double * pred);
}