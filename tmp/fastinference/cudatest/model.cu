#include <stdio.h>
#include <iostream>
#include <fstream>
#include <tuple>
#include <chrono>

#include "cuda_kernel.h"
#include "modelW.hpp"
#include "utils.cuh"

// TODO add code here

// layer 2 regular_conv2d

__global__ void layer2_gpu_kernel(int *d_cuda_layer_1_output, signed char *d_layer_2_bias, signed char *d_cuda_layer_2_weight, int *d_cuda_layer_2_output){

    int N = (28+1); // +1 to cover all edges (fixes bug #ky2)
    int kernel_size = kernel_shape[2];

    int tid = threadIdx.x; // = h
    int bid = blockIdx.y;  // = w
    int h = tid, w = bid;

    int m = blockIdx.z; // neurons in z-dir

    // batches in x-dir
    int b = blockIdx.x;
    //each block is assigned to a row of an image, iy index of y value                  
    int iy = blockIdx.y + (kernel_size - 1)/2;  
    //each thread is assigned to a pixel of a row, ix index of x value
    int ix = threadIdx.x + (kernel_size - 1)/2; 
    
    //idx global index (all blocks) of the image pixel 
    int idx = iy*N +ix;

    // bias is applied to every pixel
    if(tid < N){
        if(b < 1){
            if(m < 32) {
                d_cuda_layer_2_output[index4D_cuda(b,h,w,m,26,26,32)] = d_layer_2_bias[m];
            }
        }
    }

    __syncthreads();

    // edge pixels are skipped here because they cannot fit entire convolution window
    if(idx < N*N){
        for (int kH = 0; kH < 3; kH++) {
            for (int kW = 0; kW < 3; kW++) {
                if(b < BATCH_SIZE){
                    for (int c = 0; c < 1; c++) {
                        if(m < 32) {
                            d_cuda_layer_2_output[index4D_cuda(b,h,w,m,26,26,32)] += d_cuda_layer_2_weight[index4D_cuda(kH,kW,c,m,3,1,32)] * d_cuda_layer_1_output[index4D_cuda(b,(h * 1 + kH - 0),(w * 1 + kW - 0),c,28,28,1)];
                        }
                    }
                }
            }
        }
    }
    
}

float layer2_gpu_cuda(int * cuda_layer_1_output, int * cuda_layer_2_output){
    //setUniGPU(); // use the second GPU on Uni-server because the first is used most of the time

    // flatten 3D -> 1D arrays
    // flatten layer_2_weight
    signed char *cuda_layer_2_weight = (signed char *) layer_2_weight;

    // flatten layer_1_output -> already flattened?
    // int *cuda_layer_1_output = (int *) layer_1_output;

    // prepare for kernel call
    // declare storage on device
    int *d_cuda_layer_1_output; // storage on device for cuda_layer_1_output
    signed char *d_layer_2_bias; // storage on device for layer_2_bias
    signed char *d_cuda_layer_2_weight; // storage on device for cuda_layer_2_weight
    int *d_cuda_layer_2_output; // RESULT storage on device for cuda_layer_2_output

    // allocate GPU device buffers
    // Note: batch_size included in input and output shapes
    cudaMalloc((void **) &d_cuda_layer_1_output, 1*1*28*28*sizeof(int)); // dim of cuda_layer_1_output
    cudaMalloc((void **) &d_layer_2_bias, *sizeof(signed char)); // dim of layer_2_bias
    cudaMalloc((void **) &d_cuda_layer_2_weight, 3*3*1*32*sizeof(signed char)); // dim of layer_2_weight
    cudaMalloc((void **) &d_cuda_layer_2_output, 1*32*26*26*sizeof(int)); // dim of layer_2_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_1_output, cuda_layer_1_output, (1*1*28*28*sizeof(int)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_2_bias, layer_2_bias, (*sizeof(signed char)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cuda_layer_2_weight, cuda_layer_2_weight, (3*3*1*32*sizeof(signed char)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes
    const int BLKXSIZE = 26;
    const int BLKYSIZE = 1;
    const int BLKZSIZE = 1;
    const int GRIDXSIZE = 1;
    const int GRIDYSIZE = 26;
    const int GRIDZSIZE = 32;

    const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE, BLKZSIZE);
    const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE, GRIDZSIZE);

    // timing of the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // compute result - kernel call
    cudaEventRecord(start);
    layer2_gpu_kernel<<<numBlocks,threadsPerBlock>>>(d_cuda_layer_1_output, d_layer_2_bias, d_cuda_layer_2_weight, d_cuda_layer_2_output);
    cudaCheckErrors("Kernel launch failure");
    cudaEventRecord(stop);

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy result from device to host
    cudaMemcpy(cuda_layer_2_output, d_cuda_layer_2_output, (1*32*26*26*sizeof(int)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // free the memory
    cudaFree(d_cuda_layer_1_output);
    cudaFree(d_layer_2_bias);
    cudaFree(d_cuda_layer_2_weight);
    cudaFree(d_cuda_layer_2_output);
    cudaCheckErrors("cudaFree fail");

    // // checksum L2
    // float sum_gpu = 0;
    // ofstream gg2("layer2/par.out");
    // for(int b = 0; b < 1; b++){
    //     sum_gpu = 0;
    //     for(int i = b*26*26*32; i < (b+1)*26*26*32; i++){
    //         sum_gpu += cuda_layer_2_output[i];
    //         gg1<<cuda_layer_2_output[i]<<" ";  
    //     }
    //     cout<<fixed<<"layer 2(GPU): batch "<<b<<": "<<sum_gpu<<endl;
    // }

    return milliseconds;
}

// layer 4 maxpool
// layer 5 conv2d
// layer 7 maxpool
// layer 9 gemm
// layer 11 gemm
