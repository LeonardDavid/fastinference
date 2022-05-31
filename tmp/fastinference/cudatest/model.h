#pragma once 

namespace FAST_INFERENCE {    
    #define REF_ACCURACY 90.31

    #define N_CLASSES 10
    #define N_FEATURES 784

    std::tuple<float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float>
    // void
    predict_cudatest(int const * const x, int * pred);
}