#pragma once 

namespace FAST_INFERENCE {    
    #define REF_ACCURACY 78.8

    #define N_CLASSES 10
    #define N_FEATURES 784

    void predict_cnntest_fashion1(int const * const x, int * pred);
}