#pragma once 

namespace FAST_INFERENCE {    
    #define REF_ACCURACY 37.330000000000005

    #define N_CLASSES 10
    #define N_FEATURES 3072

    void predict_cnntest_cifar(int const * const x, int * pred);
}