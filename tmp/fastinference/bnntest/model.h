#pragma once 

namespace FAST_INFERENCE {    
    #define REF_ACCURACY 87.03

    #define N_CLASSES 10
    #define N_FEATURES 784

    void predict_bnntest(int const * const x, int * pred);
}