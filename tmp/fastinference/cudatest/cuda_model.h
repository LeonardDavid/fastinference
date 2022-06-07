#include <stdio.h>
#include "cuda_kernel.h"


float layer2_gpu(int * layer1, int * layer2){
    return layer2_gpu_cuda(layer1, layer2);
}

float layer4_gpu(unsigned int * layer1, unsigned int * layer2){
    return layer4_gpu_cuda(layer1, layer2);
}

float layer7_gpu(unsigned int * layer1, unsigned int * layer2){
    return layer7_gpu_cuda(layer1, layer2);
}
