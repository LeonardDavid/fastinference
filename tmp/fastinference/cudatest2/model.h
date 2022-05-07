#pragma once 

namespace FAST_INFERENCE {    
    #define REF_ACCURACY 88.61

    #define N_CLASSES 10
    #define N_FEATURES 784

    void predict_cudatest2(double const * const x, double * pred);
}