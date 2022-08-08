#pragma once 

namespace FAST_INFERENCE {
    #define N_CLASSES 10
    #define N_FEATURES 784

    void predict_model(int const * const x, int * pred, float * ln_times);
}