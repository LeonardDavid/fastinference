#pragma once 

namespace FAST_INFERENCE {    
    #define REF_ACCURACY 10.76

    #define N_CLASSES 10
    #define N_FEATURES 3072

    void predict_cnntest_cifar2(int const * const x, int * pred);
}