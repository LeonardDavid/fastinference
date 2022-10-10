#pragma once 

namespace FAST_INFERENCE {    
    #define REF_ACCURACY 77.24

    #define N_CLASSES 10
    #define N_FEATURES 784

    void predict_cnntest_fashion2(int const * const x, int * pred);
}