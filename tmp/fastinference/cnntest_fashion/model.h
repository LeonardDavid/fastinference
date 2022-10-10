#pragma once 

namespace FAST_INFERENCE {    
    #define REF_ACCURACY 69.38

    #define N_CLASSES 10
    #define N_FEATURES 784

    void predict_cnntest_fashion(int const * const x, int * pred);
}