#include <stdio.h>
#include <iostream>
#include <fstream>
#include <tuple>
#include <chrono>

#include "cuda_kernel.h"
#include "modelW.hpp"
#include "utils.cuh"

// TODO add code here

// layer 2 regular_conv2d x

__global__ void layer2_gpu_kernel(int *d_cuda_layer_1_output, signed char *d_layer_2_bias, signed char *d_cuda_layer_2_weight, int *d_cuda_layer_2_output){
    
    int h = threadIdx.x;
    int w = blockIdx.y * blockIdx.y + threadIdx.y;

    int b = blockIdx.x; // Batches index on x grid

    if(h < 26 && w < 26){
        if(b < 256){
            for (int m=0; m < 32; m++) {
                d_cuda_layer_2_output[index4D_cuda(b,h,w,m,26,26,32)] = d_layer_2_bias[m];
            }
        }

        for (int kH = 0; kH < 3; kH++) {
            for (int kW = 0; kW < 3; kW++) {
                if(b < 256){
                    for (int c = 0; c < 1; c++) {
                        for (int m=0; m < 32; m++) {
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
    cudaMalloc((void **) &d_cuda_layer_1_output, 256*1*1*28*28*sizeof(int)); // dim of cuda_layer_1_output
    cudaMalloc((void **) &d_layer_2_bias, 32*sizeof(signed char)); // dim of layer_2_bias
    cudaMalloc((void **) &d_cuda_layer_2_weight, 3*3*1*32*sizeof(signed char)); // dim of layer_2_weight
    cudaMalloc((void **) &d_cuda_layer_2_output, 256*1*32*26*26*sizeof(int)); // dim of layer_2_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_1_output, cuda_layer_1_output, (256*1*1*28*28*sizeof(int)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_2_bias, layer_2_bias, (32*sizeof(signed char)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cuda_layer_2_weight, cuda_layer_2_weight, (3*3*1*32*sizeof(signed char)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes
    // TODO: allow for bigger image sizes than 32x32 (threads/block limit)
    const int BLKXSIZE = 26;
    const int BLKYSIZE = 26;
    const int BLKZSIZE = 1;
    const int GRIDXSIZE = 256;
    const int GRIDYSIZE = 1;
    const int GRIDZSIZE = 1;

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
    cudaMemcpy(cuda_layer_2_output, d_cuda_layer_2_output, (256*1*32*26*26*sizeof(int)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // free the memory
    cudaFree(d_cuda_layer_1_output);
    cudaFree(d_layer_2_bias);
    cudaFree(d_cuda_layer_2_weight);
    cudaFree(d_cuda_layer_2_output);
    cudaCheckErrors("cudaFree fail");

    return milliseconds;
}

// layer 4 maxpool x

__global__ void layer4_gpu_kernel(unsigned int *d_cuda_layer_3_output, unsigned int *d_cuda_layer_4_output){

    int h = threadIdx.x; // modified for work with multiple batches
    int w = blockDim.y * blockIdx.y + threadIdx.y;

    int b = blockIdx.x; // Batches index in grid x dir
    
    if(h < 13 && w < 13){
        if(b < 256){
            for(int c = 0; c < 1; c++)
            {
                d_cuda_layer_4_output[index4D_cuda(b,h,w,c,13,13,1)] = 0;
            }
        }

        for (int kH = 0; kH < 2; kH++) {
            for (int kW = 0; kW < 2; kW++) {
                if(b < 256){
                    for(int c = 0; c < 1; c++)
                    {
                        d_cuda_layer_4_output[index4D_cuda(b,h,w,c,13,13,1)] |= d_cuda_layer_3_output[index4D_cuda(b,(h * 2 + kH),(w * 2 + kW),c,26,26,32)];
                    }
                }
            }
        }
    }
}

float layer4_gpu_cuda(unsigned int * cuda_layer_3_output, unsigned int * cuda_layer_4_output){
    //setUniGPU(); // use the second GPU on Uni-server because the first is used most of the time

    // flatten 3D -> 1D arrays
    // no 3D arrays to be flattened

    // prepare for kernel call
    // declare storage on device
    unsigned int *d_cuda_layer_3_output; // storage on device for cuda_layer_3_output
    unsigned int *d_cuda_layer_4_output; // RESULT storage on device for cuda_layer_4_output

    // allocate GPU device buffers
    // Note: batch_size included in input and output shapes
    cudaMalloc((void **) &d_cuda_layer_3_output, 256*1*32*26*26*sizeof(unsigned int)); // dim of cuda_layer_3_output
    cudaMalloc((void **) &d_cuda_layer_4_output, 256*1*32*13*13*sizeof(unsigned int)); // dim of layer_4_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_3_output, cuda_layer_3_output, (256*1*32*26*26*sizeof(unsigned int)), cudaMemcpyHostToDevice);

    // define thread and block sizes
    // TODO: allow for bigger image sizes than 32x32 (threads/block limit)
    const int BLKXSIZE = 13;
    const int BLKYSIZE = 13;
    const int BLKZSIZE = 1;
    const int GRIDXSIZE = 256;
    const int GRIDYSIZE = 1;
    const int GRIDZSIZE = 1;

    const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE, BLKZSIZE);
    const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE, GRIDZSIZE);

    // timing of the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // compute result - kernel call
    cudaEventRecord(start);
    layer4_gpu_kernel<<<numBlocks,threadsPerBlock>>>(d_cuda_layer_3_output, d_cuda_layer_4_output);
    cudaCheckErrors("Kernel launch failure");
    cudaEventRecord(stop);

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy result from device to host
    cudaMemcpy(cuda_layer_4_output, d_cuda_layer_4_output, (256*1*32*13*13*sizeof(unsigned int)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // free the memory
    cudaFree(d_cuda_layer_3_output);
    cudaFree(d_cuda_layer_4_output);
    cudaCheckErrors("cudaFree fail");

    return milliseconds;
}

// layer 5 conv2d x

__global__ void layer5_gpu_kernel(unsigned int *d_cuda_layer_4_output, signed char *d_layer_5_bias, unsigned int *d_cuda_layer_5_weight, signed int *d_cuda_layer_5_output){
    
    int h = threadIdx.x;
    int w = blockIdx.y * blockIdx.y + threadIdx.y;

    int b = blockIdx.x; // Batches index on x grid

    if(h < 11 && w < 11){
        if(b < 256){
            for (int m=0; m < 32; m++) {
                d_cuda_layer_5_output[index4D_cuda(b,h,w,m,11,11,32)] = d_layer_5_bias[m];
            }
        }
        for (int kH = 0; kH < 3; kH++) {
            for (int kW = 0; kW < 3; kW++) {
                if(b < 256){
                    for (int c = 0; c < 1; c++) {
                        for (int m=0; m < 32; m++) {
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
    // Note: batch_size included in input and output shapes
    cudaMalloc((void **) &d_cuda_layer_4_output, 256*1*32*13*13*sizeof(unsigned int)); // dim of cuda_layer_4_output
    cudaMalloc((void **) &d_layer_5_bias, 32*sizeof(signed char)); // dim of layer_5_bias
    cudaMalloc((void **) &d_cuda_layer_5_weight, 3*3*32*32*sizeof(unsigned int)); // dim of layer_5_weight
    cudaMalloc((void **) &d_cuda_layer_5_output, 256*1*32*11*11*sizeof(signed int)); // dim of layer_5_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_4_output, cuda_layer_4_output, (256*1*32*13*13*sizeof(unsigned int)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_5_bias, layer_5_bias, (32*sizeof(signed char)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cuda_layer_5_weight, cuda_layer_5_weight, (3*3*32*32*sizeof(unsigned int)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes
    // TODO: allow for bigger image sizes than 32x32 (threads/block limit)
    const int BLKXSIZE = 11;
    const int BLKYSIZE = 11;
    const int BLKZSIZE = 1;
    const int GRIDXSIZE = 256;
    const int GRIDYSIZE = 1;
    const int GRIDZSIZE = 1;

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
    cudaMemcpy(cuda_layer_5_output, d_cuda_layer_5_output, (256*1*32*11*11*sizeof(signed int)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // free the memory
    cudaFree(d_cuda_layer_4_output);
    cudaFree(d_layer_5_bias);
    cudaFree(d_cuda_layer_5_weight);
    cudaFree(d_cuda_layer_5_output);
    cudaCheckErrors("cudaFree fail");

    return milliseconds;
}
// layer 7 maxpool x

__global__ void layer7_gpu_kernel(unsigned int *d_cuda_layer_6_output, unsigned int *d_cuda_layer_7_output){

    int h = threadIdx.x; // modified for work with multiple batches
    int w = blockDim.y * blockIdx.y + threadIdx.y;

    int b = blockIdx.x; // Batches index in grid x dir
    
    if(h < 5 && w < 5){
        if(b < 256){
            for(int c = 0; c < 1; c++)
            {
                d_cuda_layer_7_output[index4D_cuda(b,h,w,c,5,5,1)] = 0;
            }
        }

        for (int kH = 0; kH < 2; kH++) {
            for (int kW = 0; kW < 2; kW++) {
                if(b < 256){
                    for(int c = 0; c < 1; c++)
                    {
                        d_cuda_layer_7_output[index4D_cuda(b,h,w,c,5,5,1)] |= d_cuda_layer_6_output[index4D_cuda(b,(h * 2 + kH),(w * 2 + kW),c,11,11,32)];
                    }
                }
            }
        }
    }
}

float layer7_gpu_cuda(unsigned int * cuda_layer_6_output, unsigned int * cuda_layer_7_output){
    //setUniGPU(); // use the second GPU on Uni-server because the first is used most of the time

    // flatten 3D -> 1D arrays
    // no 3D arrays to be flattened

    // prepare for kernel call
    // declare storage on device
    unsigned int *d_cuda_layer_6_output; // storage on device for cuda_layer_6_output
    unsigned int *d_cuda_layer_7_output; // RESULT storage on device for cuda_layer_7_output

    // allocate GPU device buffers
    // Note: batch_size included in input and output shapes
    cudaMalloc((void **) &d_cuda_layer_6_output, 256*1*32*11*11*sizeof(unsigned int)); // dim of cuda_layer_6_output
    cudaMalloc((void **) &d_cuda_layer_7_output, 256*1*32*5*5*sizeof(unsigned int)); // dim of layer_7_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_6_output, cuda_layer_6_output, (256*1*32*11*11*sizeof(unsigned int)), cudaMemcpyHostToDevice);

    // define thread and block sizes
    // TODO: allow for bigger image sizes than 32x32 (threads/block limit)
    const int BLKXSIZE = 5;
    const int BLKYSIZE = 5;
    const int BLKZSIZE = 1;
    const int GRIDXSIZE = 256;
    const int GRIDYSIZE = 1;
    const int GRIDZSIZE = 1;

    const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE, BLKZSIZE);
    const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE, GRIDZSIZE);

    // timing of the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // compute result - kernel call
    cudaEventRecord(start);
    layer7_gpu_kernel<<<numBlocks,threadsPerBlock>>>(d_cuda_layer_6_output, d_cuda_layer_7_output);
    cudaCheckErrors("Kernel launch failure");
    cudaEventRecord(stop);

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy result from device to host
    cudaMemcpy(cuda_layer_7_output, d_cuda_layer_7_output, (256*1*32*5*5*sizeof(unsigned int)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // free the memory
    cudaFree(d_cuda_layer_6_output);
    cudaFree(d_cuda_layer_7_output);
    cudaCheckErrors("cudaFree fail");

    return milliseconds;
}

// layer 9 gemm x

__global__ void layer9_gpu_kernel(unsigned int *d_cuda_layer_8_output, signed char *d_layer_9_bias, unsigned int *d_cuda_layer_9_weight, signed int *d_cuda_layer_9_output){

    int z = blockDim.x * blockIdx.z + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int d = z*blockDim.x+y;

    int b = blockIdx.x;

    if(d < 32){
        if(b < 256){
            d_cuda_layer_9_output[b * 32 + d] = d_layer_9_bias[d];
            for(int i = 0; i < 25; i++){
                d_cuda_layer_9_output[b*32 + d] += 2 * __popc((unsigned int)~(unsigned int)(d_cuda_layer_9_weight[d*25 + i] ^ d_cuda_layer_8_output[b*25 + i])) - 32;
            }
        }
    }   
}

float layer9_gpu_cuda(unsigned int * cuda_layer_8_output, signed int * cuda_layer_9_output){

    // setUniGPU(); // use the second GPU on Uni-server because the first is used most of the time

    // flatten 3D -> 1D arrays
    // flatten layer_9_weight
    unsigned int *cuda_layer_9_weight = (unsigned int *) layer_9_weight;

    // prepare for kernel call
    // declare storage on device
    unsigned int *d_cuda_layer_8_output; // storage on device for cuda_layer_8_output
    signed char *d_layer_9_bias; // storage on device for layer_9_bias
    unsigned int *d_cuda_layer_9_weight; // storage on device for cuda_layer_9_weight
    signed int *d_cuda_layer_9_output; // RESULT storage on device for cuda_layer_9_output

    // allocate GPU device buffers
    // Note: batch_size included in input and output shapes
    cudaMalloc((void **) &d_cuda_layer_8_output, 256*1*800*sizeof(unsigned int)); // dim of cuda_layer_8_output
    cudaMalloc((void **) &d_layer_9_bias, 32*sizeof(signed char)); // dim of layer_9_bias
    cudaMalloc((void **) &d_cuda_layer_9_weight, 32*800*sizeof(unsigned int)); // dim of layer_9_weight
    cudaMalloc((void **) &d_cuda_layer_9_output, 256*1*32*sizeof(signed int)); // dim of layer_9_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_8_output, cuda_layer_8_output, (256*1*800*sizeof(unsigned int)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_9_bias, layer_9_bias, (32*sizeof(signed char)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cuda_layer_9_weight, cuda_layer_9_weight, (32*800*sizeof(unsigned int)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes
    /*
        Maximum threads in a block: 1024 => Maximum block size 32x32
        if more than 1024 threads are needed, then set block size to maximum (32x32) and put multiple blocks in z-dir
        else if less than 1024 are needed, then only create 1 (square) block in z-dir, of size ceil(sqrt(THREADS_NEEDED))
    */
    const int BLKXSIZE = std::ceil(sqrt(32));
    const int BLKYSIZE = std::ceil(sqrt(32));
    const int BLKZSIZE = 1;
    const int GRIDXSIZE = 256;
    const int GRIDYSIZE = 1;
    const int GRIDZSIZE = 1;

    const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE, BLKZSIZE);
    const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE, GRIDZSIZE);

    // timing of the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // compute result - kernel call
    cudaEventRecord(start);
    layer9_gpu_kernel<<<numBlocks,threadsPerBlock>>>(d_cuda_layer_8_output, d_layer_9_bias, d_cuda_layer_9_weight, d_cuda_layer_9_output);
    cudaCheckErrors("Kernel launch failure");
    cudaEventRecord(stop);

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy result from device to host
    cudaMemcpy(cuda_layer_9_output, d_cuda_layer_9_output, (256*1*32*sizeof(signed int)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // free the memory
    cudaFree(d_cuda_layer_8_output);
    cudaFree(d_layer_9_bias);
    cudaFree(d_cuda_layer_9_weight);
    cudaFree(d_cuda_layer_9_output);
    cudaCheckErrors("cudaFree fail");

    return milliseconds;
}
// layer 11 gemm x

__global__ void layer11_gpu_kernel(unsigned int *d_cuda_layer_10_output, signed char *d_layer_11_bias, unsigned int *d_cuda_layer_11_weight, signed int *d_cuda_layer_11_output){

    int z = blockDim.x * blockIdx.z + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int d = z*blockDim.x+y;

    int b = blockIdx.x;

    if(d < 10){
        if(b < 256){
            d_cuda_layer_11_output[b * 10 + d] = d_layer_11_bias[d];
            for(int i = 0; i < 1; i++){
                d_cuda_layer_11_output[b*10 + d] += 2 * __popc((unsigned int)~(unsigned int)(d_cuda_layer_11_weight[d*1 + i] ^ d_cuda_layer_10_output[b*1 + i])) - 32;
            }
        }
    }   
}

float layer11_gpu_cuda(unsigned int * cuda_layer_10_output, signed int * cuda_layer_11_output){

    // setUniGPU(); // use the second GPU on Uni-server because the first is used most of the time

    // flatten 3D -> 1D arrays
    // flatten layer_11_weight
    unsigned int *cuda_layer_11_weight = (unsigned int *) layer_11_weight;

    // prepare for kernel call
    // declare storage on device
    unsigned int *d_cuda_layer_10_output; // storage on device for cuda_layer_10_output
    signed char *d_layer_11_bias; // storage on device for layer_11_bias
    unsigned int *d_cuda_layer_11_weight; // storage on device for cuda_layer_11_weight
    signed int *d_cuda_layer_11_output; // RESULT storage on device for cuda_layer_11_output

    // allocate GPU device buffers
    // Note: batch_size included in input and output shapes
    cudaMalloc((void **) &d_cuda_layer_10_output, 256*1*32*sizeof(unsigned int)); // dim of cuda_layer_10_output
    cudaMalloc((void **) &d_layer_11_bias, 10*sizeof(signed char)); // dim of layer_11_bias
    cudaMalloc((void **) &d_cuda_layer_11_weight, 10*32*sizeof(unsigned int)); // dim of layer_11_weight
    cudaMalloc((void **) &d_cuda_layer_11_output, 256*1*10*sizeof(signed int)); // dim of layer_11_output
    cudaCheckErrors("Failed to allocate device buffer");

    // copy input data from host on device
    cudaMemcpy(d_cuda_layer_10_output, cuda_layer_10_output, (256*1*32*sizeof(unsigned int)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_11_bias, layer_11_bias, (10*sizeof(signed char)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cuda_layer_11_weight, cuda_layer_11_weight, (10*32*sizeof(unsigned int)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    // define thread and block sizes
    /*
        Maximum threads in a block: 1024 => Maximum block size 32x32
        if more than 1024 threads are needed, then set block size to maximum (32x32) and put multiple blocks in z-dir
        else if less than 1024 are needed, then only create 1 (square) block in z-dir, of size ceil(sqrt(THREADS_NEEDED))
    */
    const int BLKXSIZE = std::ceil(sqrt(10));
    const int BLKYSIZE = std::ceil(sqrt(10));
    const int BLKZSIZE = 1;
    const int GRIDXSIZE = 256;
    const int GRIDYSIZE = 1;
    const int GRIDZSIZE = 1;

    const dim3 threadsPerBlock(BLKXSIZE, BLKYSIZE, BLKZSIZE);
    const dim3 numBlocks(GRIDXSIZE, GRIDYSIZE, GRIDZSIZE);

    // timing of the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // compute result - kernel call
    cudaEventRecord(start);
    layer11_gpu_kernel<<<numBlocks,threadsPerBlock>>>(d_cuda_layer_10_output, d_layer_11_bias, d_cuda_layer_11_weight, d_cuda_layer_11_output);
    cudaCheckErrors("Kernel launch failure");
    cudaEventRecord(stop);

    // synchronize threads
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA synchronize failure");
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy result from device to host
    cudaMemcpy(cuda_layer_11_output, d_cuda_layer_11_output, (256*1*10*sizeof(signed int)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");

    // free the memory
    cudaFree(d_cuda_layer_10_output);
    cudaFree(d_layer_11_bias);
    cudaFree(d_cuda_layer_11_weight);
    cudaFree(d_cuda_layer_11_output);
    cudaCheckErrors("cudaFree fail");

    return milliseconds;
}
