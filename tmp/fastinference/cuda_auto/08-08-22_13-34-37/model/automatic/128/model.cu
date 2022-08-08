#include <stdio.h>
#include <iostream>
#include <fstream>
#include <tuple>
#include <chrono>

#include "cuda_kernel.h"
#include "modelW.hpp"
#include "utils.cuh"

// TODO add code here

// layer 2 regular_conv2d cpu
// layer 4 maxpool cpu
// layer 5 conv2d xyz

__global__ void layer5_gpu_kernel(unsigned int *d_cuda_layer_4_output, signed char *d_layer_5_bias, unsigned int *d_cuda_layer_5_weight, signed int *d_cuda_layer_5_output){
    
    int N = (13+1); // +1 to cover all edges (fixes bug #ky2)
    int kernel_size = 3;

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
        if(b < 128){
            if(m < 32) {
                d_cuda_layer_5_output[index4D_cuda(b,h,w,m,11,11,32)] = d_layer_5_bias[m];
            }
        }
    }

    __syncthreads();

    if(idx < N*N){
        for (int kH = 0; kH < 3; kH++) {
            for (int kW = 0; kW < 3; kW++) {
                if(b < 128){
                    for (int c = 0; c < 1; c++) {
                        if(m < 32) {
                            d_cuda_layer_5_output[index4D_cuda(b,h,w,m,11,11,32)] += 2 * __popc((unsigned int)~(unsigned int)(d_cuda_layer_5_weight[index4D_cuda(kH,kW,m,c,3,32,1)] ^ d_cuda_layer_4_output[index4D_cuda(b,(h * 1 + kH - 0),(w * 1 + kW - 0),c,13,13,1)])) - 32;
                        }
                    }
                }
            }
        }
    }
}

float layer5_gpu_cuda(unsigned int * cuda_layer_4_output, signed int * cuda_layer_5_output){

    //setUniGPU(); // use the second GPU on Uni-server because the first is used most of the time

    // flatten 3D -> 1D arrays
    // flatten layer_5_weight
    unsigned int *cuda_layer_5_weight = (unsigned int *) layer_5_weight;

    // prepare for kernel call
    // declare storage on device
    unsigned int *d_cuda_layer_4_output; // storage on device for cuda_layer_4_output
    signed char *d_layer_5_bias; // storage on device for layer_5_bias
    unsigned int *d_cuda_layer_5_weight; // storage on device for cuda_layer_5_weight
    signed int *d_cuda_layer_5_output; // RESULT storage on device for cuda_layer_5_output

    // allocate GPU device buffers
    cudaMalloc((void **) &d_cuda_layer_4_output, 128*1*32*13*13*sizeof(unsigned int)); // dim of cuda_layer_4_output
    cudaMalloc((void **) &d_layer_5_bias, 32*sizeof(signed char)); // dim of layer_5_bias
    cudaMalloc((void **) &d_cuda_layer_5_weight, 3*3*32*32*sizeof(unsigned int)); // dim of layer_5_weight
    cudaMalloc((void **) &d_cuda_layer_5_output, 128*1*32*11*11*sizeof(signed int)); // dim of layer_5_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_4_output, cuda_layer_4_output, (128*1*32*13*13*sizeof(unsigned int)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_5_bias, layer_5_bias, (32*sizeof(signed char)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cuda_layer_5_weight, cuda_layer_5_weight, (3*3*32*32*sizeof(unsigned int)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes
    const int BLKXSIZE = 11;
    const int BLKYSIZE = 1;
    const int BLKZSIZE = 1;
    const int GRIDXSIZE = 128;
    const int GRIDYSIZE = 11;
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
    layer5_gpu_kernel<<<numBlocks,threadsPerBlock>>>(d_cuda_layer_4_output, d_layer_5_bias, d_cuda_layer_5_weight, d_cuda_layer_5_output);
    cudaCheckErrors("Kernel launch failure");
    cudaEventRecord(stop);

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy result from device to host
    cudaMemcpy(cuda_layer_5_output, d_cuda_layer_5_output, (128*1*32*11*11*sizeof(signed int)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // free the memory
    cudaFree(d_cuda_layer_4_output);
    cudaFree(d_layer_5_bias);
    cudaFree(d_cuda_layer_5_weight);
    cudaFree(d_cuda_layer_5_output);
    cudaCheckErrors("cudaFree fail");

    return milliseconds;
}
// layer 7 maxpool cpu
// layer 9 gemm cpu
// layer 11 gemm cpu
