#include <stdio.h>
#include "cuda_kernel.h"


float layer2_gpu(int * layer1, int * layer2){
    return layer2_gpu_cuda(layer1, layer2);
}
