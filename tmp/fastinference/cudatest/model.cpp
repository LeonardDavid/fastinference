#include <math.h>
#include <stdint.h>
#include <limits>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <chrono>

#include "utils.h"
#include "modelW.hpp"
#include "cuda_model.h"

using namespace std;

namespace FAST_INFERENCE {

std::tuple<float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float>
//void
predict_cudatest(int const * const x, int * pred) {
    auto layer_0_output = x;

    float kernel_time = 0;
    float sum_cpu = 0, sum_gpu = 0;
	  
    // Layer 1: Reshape
    auto layer_1_output = (int (*)[28][1]) layer_0_output;

    // flatten layer_1_output for CUDA functions
    int *cuda_layer_1_output = (int *) layer_1_output;

    // variables for compatibility
    float l1_time = 0;
    float l1_kernel_time = 0;

    // // Layer 2: regular_conv2d
    // auto start = std::chrono::high_resolution_clock::now();
    // // for(int b = 0; b < 2; b++){
    // 	for (int h = 0; h < 26; h++) {
    // 		for (int w = 0; w < 26; w++) {
    // 			for (int m = 0; m < 32; m++) {
    // 				//layer_2_output[b][h][w][m] = layer_2_bias[m];
    //  				cuda_layer_2_output[index4D(b,h,w,m,26,26,32)] = layer_2_bias[m];
    // 			}
    // 			for (int kH = 0; kH < 3; kH++) {
    // 				for (int kW = 0; kW < 3; kW++) {
    // 					for (int c = 0; c < 1; c++) {
    // 						for (int m = 0; m < 32; m++) {
    //               // not sure if [b] needed in layer_1_output[?][h * 1 + kH - 0][w * 1 + kW - 0][c];
    // 							//layer_2_output[b][h][w][m] += layer_2_weight[kH][kW][c][m] * layer_1_output[h * 1 + kH - 0][w * 1 + kW - 0][c];
    // 							cuda_layer_2_output[index4D(b,h,w,m,26,26,32)] += layer_2_weight[kH][kW][c][m] * layer_1_output[h * 1 + kH - 0][w * 1 + kW - 0][c];
    // 						}
    // 					}
    // 				}
    // 			}
    // 		}
    // 	}
    // }
    // // auto end = std::chrono::high_resolution_clock::now();
    // auto l2_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
    // float l2_kernel_time = 0;

    // // checksum L2 = 
    // ofstream g2("outputs/layer2/orig.out");
    // for(int b = 0; b < 2; b++){
    //   sum_cpu = 0;
    //   for (int h = 0; h < 26; h++) {// 
    //     for (int w = 0; w < 26; w++) {
    //       for (int m = 0; m < 32; m++) {
    //         sum_cpu += cuda_layer_2_output[index4D(b,h,w,m,26,26,32)];
    //         g2<<cuda_layer_2_output[index4D(b,h,w,m,26,26,32)]<<" ";  
    //       }
    //     }
    //   }
    //   cout<<fixed<<"layer 2(CPU): batch "<<b<<": "<<sum_cpu<<endl;
    // }
    // cout<<endl;

    /* Layer 2 GPU */
    auto start = std::chrono::high_resolution_clock::now();
    kernel_time += layer2_gpu(cuda_layer_1_output, cuda_layer_2_output);
    auto end = std::chrono::high_resolution_clock::now();
    auto l2_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
    float l2_kernel_time = kernel_time;
    l2_time -= l2_kernel_time*1000000.0f; // ms->ns

    // // checksum L2 = 
    // ofstream gg2("outputs/layer2/par.out");
    // for(int b = 0; b < 2; b++){
    //   sum_gpu = 0;
    //   for(int i=b*26*26*32;i<(b+1)*26*26*32;i++){
    //     sum_gpu += cuda_layer_2_output[i];
    //     gg2<<cuda_layer_2_output[i]<<" ";  
    //   }
    //   cout<<fixed<<"layer 2(GPU): batch "<<b<<": "<<sum_gpu<<endl;
    // }
    // cout<<endl;

    // Layer 3: Step
    start = std::chrono::high_resolution_clock::now();
    for(int b = 0; b < 2; b++){
      for (int h = 0; h < 26; h++) {
        for (int w = 0; w < 26; w++) {
          for (int c = 0; c < 32; c++) {
            // if (layer_2_output[b][h][w][c] >= layer_3_threshold[c]) 
            if (cuda_layer_2_output[index4D(b,h,w,c,26,26,32)] >= layer_3_threshold[c]) 
            { 
              layer_3_output[b][h][w][c / 32] |= (1U << (31 - c % 32));
            } else {
              layer_3_output[b][h][w][c / 32] &= ~(1U << (31 - c % 32));
            }
          }
        }
      }
    }

    // needed for GPU, but on CPU layers work with the normal layer_x_output from step
    for(int b = 0; b < 2; b++){
      for (int h = 0; h < 26; h++) {
        for (int w = 0; w < 26; w++) {
          for (int c = 0; c < 1; c++) {
            cuda_layer_3_output[index4D(b,h,w,c,26,26,32)] = layer_3_output[b][h][w][c];
          }
        }
      }
    }

    end = std::chrono::high_resolution_clock::now();
    auto l3_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());  
    float l3_kernel_time = 0;

    // // Layer 4: MaxPool
    // start = std::chrono::high_resolution_clock::now();
    // // for(int b = 0; b < 2; b++){
    //   for (int h = 0; h < 13; h++) {
    //     for (int w = 0; w < 13; w++) {
    //       for (int c = 0; c < 1; c++) {
    //         cuda_layer_4_output[index4D(b,h,w,c,13,13,1)] = 0;
    //       }
    //       for (int kH = 0; kH < 2; kH++) {
    //         for (int kW = 0; kW < 2; kW++) {
    //           for (int c = 0; c < 1; c++) {
    //             //             cuda_layer_4_output[index4D(b,h,w,c,13,13,1)] |= layer_3_output[b][h * 2 + kH][w * 2 + kW][c];
    //             //           }
    //         }
    //       }
    //     }
    //   }
    // }
    // // end = std::chrono::high_resolution_clock::now();
    // auto l4_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());  
    // float l4_kernel_time = 0;
     
    // // checksum L4 = 
    // ofstream g4("outputs/layer4/orig.out");
    // for(int b = 0; b < 2; b++){
    //   sum_cpu = 0;
    //   for (int h = 0; h < 13; h++) {
    //     for (int w = 0; w < 13; w++) {
    //       //       for (int c = 0; c < 1; c++)
    //       //       {
    //         sum_cpu += cuda_layer_4_output[index4D(b,h,w,c,13,13,1)];
    //         g4<<cuda_layer_4_output[index4D(b,h,w,c,13,13,1)]<<" ";  
    //       }
    //     }
    //   }
    //   cout<<fixed<<"layer 4(CPU): batch "<<b<<": "<<sum_cpu<<endl;
    // }
    // cout<<endl;

    /* Layer 4 GPU */ 
    start = std::chrono::high_resolution_clock::now();
    kernel_time += layer4_gpu(cuda_layer_3_output, cuda_layer_4_output);
    end = std::chrono::high_resolution_clock::now();  
    auto l4_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
    //TODO for loop at the end? float l4_kernel_time = kernel_time-l1_kernel_time-...;
    // l4_time -= l4_kernel_time*1000000.0f; // ms->ns
    float l4_kernel_time = 0;

    // // checksum L4 = 
    // ofstream gg4("outputs/layer4/par.out");
    // for(int b = 0; b < 2; b++){
    //   sum_gpu = 0;
    //   //   for(int i=b*13*13*1;i<(b+1)*13*13*1;i++)
    //   //   {
    //     sum_gpu += cuda_layer_4_output[i];
    //     gg4<<cuda_layer_4_output[i]<<" ";  
    //   }
    //   cout<<fixed<<"layer 4(GPU): batch "<<b<<": "<<sum_gpu<<endl;
    // }
    // cout<<endl;

    // // Layer 5: conv2d
    // start = std::chrono::high_resolution_clock::now();
    // // for(int b = 0; b < 2; b++){
    //   for (int h = 0; h < 11; h++) {
    //     for (int w = 0; w < 11; w++) {
    //       for (int m = 0; m < 32; m++) {
    //         cuda_layer_5_output[index4D(b,h,w,m,11,11,32)] = layer_5_bias[m];
    //       }
    //       for (int kH = 0; kH < 3; kH++) {
    //         for (int kW = 0; kW < 3; kW++) {
    //           for (int m = 0; m < 32; m++) {
    //             for (int c = 0; c < 1; c++) {
    //               //               cuda_layer_5_output[index4D(b,h,w,m,11,11,32)] += 2 * __builtin_popcount((unsigned int)~(unsigned int)(layer_5_weight[kH][kW][m][c] ^ cuda_layer_4_output[index4D(b,(h * 1 + kH - 0),(w * 1 + kW - 0),c,13,13,1)])) - 32;
    //               //             }
    //           }
    //         }
    //       }
    //     }
    //   }
    // }
    // // end = std::chrono::high_resolution_clock::now();
    // auto l5_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());  
    // float l5_kernel_time = 0;
    // 
    // // checksum L5 = 
    // ofstream g5("outputs/layer5/orig.out");
    // for(int b = 0; b < 2; b++){
    //   sum_cpu = 0;
    //   for (int h = 0; h < 11; h++) {// 
    //     for (int w = 0; w < 11; w++) {
    //       for (int m = 0; m < 32; m++) {
    //         sum_cpu += cuda_layer_5_output[index4D(b,h,w,m,11,11,32)];
    //         g5<<cuda_layer_5_output[index4D(b,h,w,m,11,11,32)]<<" ";  
    //       }
    //     }
    //   }
    //   cout<<fixed<<"layer 5(CPU): batch "<<b<<": "<<sum_cpu<<endl;
    // }
    // cout<<endl;

    /* Layer 5 GPU */ 
    start = std::chrono::high_resolution_clock::now();
    kernel_time += layer5_gpu(cuda_layer_4_output, cuda_layer_5_output);
    end = std::chrono::high_resolution_clock::now();  
    auto l5_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
    //TODO for loop at the end? float l5_kernel_time = kernel_time-l1_kernel_time-...;
    //l5_time -= l5_kernel_time*1000000.0f; // ms->ns
    float l5_kernel_time = 0;

    // // checksum L5 = 
    // ofstream gg5("outputs/layer5/par.out");
    // for(int b = 0; b < 2; b++){
    //   sum_gpu = 0;
    //   for(int i=b*11*11*32;i<(b+1)*11*11*32;i++){
    //     sum_gpu += cuda_layer_5_output[i];
    //     gg5<<cuda_layer_5_output[i]<<" ";  
    //   }
    //   cout<<fixed<<"layer 5(GPU): batch "<<b<<": "<<sum_gpu<<endl;
    // }
    // cout<<endl;

    // Layer 6: Step
    start = std::chrono::high_resolution_clock::now();
    for(int b = 0; b < 2; b++){
      for (int h = 0; h < 11; h++) {
        for (int w = 0; w < 11; w++) {
          for (int c = 0; c < 32; c++) {
            // if (layer_5_output[b][h][w][c] >= layer_6_threshold[c]) 
            if (cuda_layer_5_output[index4D(b,h,w,c,11,11,32)] >= layer_6_threshold[c]) 
            { 
              layer_6_output[b][h][w][c / 32] |= (1U << (31 - c % 32));
            } else {
              layer_6_output[b][h][w][c / 32] &= ~(1U << (31 - c % 32));
            }
          }
        }
      }
    }

    // needed for GPU, but on CPU layers work with the normal layer_x_output from step
    for(int b = 0; b < 2; b++){
      for (int h = 0; h < 11; h++) {
        for (int w = 0; w < 11; w++) {
          for (int c = 0; c < 1; c++) {
            cuda_layer_6_output[index4D(b,h,w,c,11,11,32)] = layer_6_output[b][h][w][c];
          }
        }
      }
    }

    end = std::chrono::high_resolution_clock::now();
    auto l6_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());  
    float l6_kernel_time = 0;

    // // Layer 7: MaxPool
    // start = std::chrono::high_resolution_clock::now();
    // // for(int b = 0; b < 2; b++){
    //   for (int h = 0; h < 5; h++) {
    //     for (int w = 0; w < 5; w++) {
    //       for (int c = 0; c < 1; c++) {
    //         cuda_layer_7_output[index4D(b,h,w,c,5,5,1)] = 0;
    //       }
    //       for (int kH = 0; kH < 2; kH++) {
    //         for (int kW = 0; kW < 2; kW++) {
    //           for (int c = 0; c < 1; c++) {
    //             //             cuda_layer_7_output[index4D(b,h,w,c,5,5,1)] |= layer_6_output[b][h * 2 + kH][w * 2 + kW][c];
    //             //           }
    //         }
    //       }
    //     }
    //   }
    // }
    // // end = std::chrono::high_resolution_clock::now();
    // auto l7_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());  
    // float l7_kernel_time = 0;
     
    // // checksum L7 = 
    // ofstream g7("outputs/layer7/orig.out");
    // for(int b = 0; b < 2; b++){
    //   sum_cpu = 0;
    //   for (int h = 0; h < 5; h++) {
    //     for (int w = 0; w < 5; w++) {
    //       //       for (int c = 0; c < 1; c++)
    //       //       {
    //         sum_cpu += cuda_layer_7_output[index4D(b,h,w,c,5,5,1)];
    //         g7<<cuda_layer_7_output[index4D(b,h,w,c,5,5,1)]<<" ";  
    //       }
    //     }
    //   }
    //   cout<<fixed<<"layer 7(CPU): batch "<<b<<": "<<sum_cpu<<endl;
    // }
    // cout<<endl;

    /* Layer 7 GPU */ 
    start = std::chrono::high_resolution_clock::now();
    kernel_time += layer7_gpu(cuda_layer_6_output, cuda_layer_7_output);
    end = std::chrono::high_resolution_clock::now();  
    auto l7_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
    //TODO for loop at the end? float l7_kernel_time = kernel_time-l1_kernel_time-...;
    // l7_time -= l7_kernel_time*1000000.0f; // ms->ns
    float l7_kernel_time = 0;

    // // checksum L7 = 
    // ofstream gg7("outputs/layer7/par.out");
    // for(int b = 0; b < 2; b++){
    //   sum_gpu = 0;
    //   //   for(int i=b*5*5*1;i<(b+1)*5*5*1;i++)
    //   //   {
    //     sum_gpu += cuda_layer_7_output[i];
    //     gg7<<cuda_layer_7_output[i]<<" ";  
    //   }
    //   cout<<fixed<<"layer 7(GPU): batch "<<b<<": "<<sum_gpu<<endl;
    // }
    // cout<<endl;

    // Layer 8: Reshape
    // auto layer_8_output = (unsigned int (*)) layer_7_output;
    auto cuda_layer_8_output = (unsigned int (*)) cuda_layer_7_output;

    // variables for compatibility
    float l8_time = 0;
    float l8_kernel_time = 0;

    // // Layer 9: Gemm
    // start = std::chrono::high_resolution_clock::now();
    //  // for(int b = 0; b < 2; b++){
    //   for (int d = 0; d < 32; d++) {
    //     cuda_layer_9_output[b*32 + d] = layer_9_bias[d];
    //   }
    //   for (int d = 0; d < 32; d++) {
    //     for (int i = 0; i < 25; i++) {
    //       cuda_layer_9_output[b*32 + d] += 2 * __builtin_popcount((unsigned int)~(unsigned int)(layer_9_weight[d][i] ^ cuda_layer_8_output[b*25 + i])) - 32;
    //     }
    //   }
    // }
    // // end = std::chrono::high_resolution_clock::now();
    // auto l9_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());  
    // float l9_kernel_time = 0;
    // 
    // // checksum L9 
    // ofstream g9("outputs/layer9/orig.out");
    // for(int b = 0; b < 2; b++){
    //   sum_cpu = 0;
    //   for (int d = 0; d < 32; d++) {
    //     sum_cpu += cuda_layer_9_output[b*32 + d];
    //     g9<<cuda_layer_9_output[b*32 + d]<<" ";  
    //   }
    //   cout<<fixed<<"layer 9(CPU): batch "<<b<<": "<<sum_cpu<<endl;
    // }
    // cout<<endl;

    /* Layer 9 GPU */ 
    start = std::chrono::high_resolution_clock::now();
    kernel_time += layer9_gpu(cuda_layer_8_output, cuda_layer_9_output);
    end = std::chrono::high_resolution_clock::now();  
    auto l9_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
    //TODO for loop at the end? float l9_kernel_time = kernel_time-l1_kernel_time-...;
    //l9_time -= l9_kernel_time*1000000.0f; // ms->n
    float l9_kernel_time = 0;

    // // checksum L9
    // ofstream gg9("outputs/layer9/par.out");
    // for(int b = 0; b < 2; b++){
    //   sum_gpu = 0;
    //   for(int i=b*32;i<(b+1)*32;i++){
    //     sum_gpu += cuda_layer_9_output[i];
    //     gg9<<cuda_layer_9_output[i]<<" ";  
    //   }
    //   cout<<fixed<<"layer 9(GPU): batch "<<b<<": "<<sum_gpu<<endl;
    // }
    // cout<<endl;

    // Layer 10: Step
    start = std::chrono::high_resolution_clock::now();
    for(int b = 0; b < 2; b++){
      for (int d = 0; d < 32; d++) {
        if (cuda_layer_9_output[b*32 + d] >= layer_10_threshold[d]) {
          layer_10_output[b][d / 32] |= (1U << (31 - d % 32));
        } else {
          layer_10_output[b][d / 32] &= ~(1U << (31 - d % 32));
        }
      }
    }

    unsigned int *cuda_layer_10_output = (unsigned int *) layer_10_output;

    end = std::chrono::high_resolution_clock::now();
    auto l10_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());  
    float l10_kernel_time = 0;

    // // Layer 11: Gemm
    // start = std::chrono::high_resolution_clock::now();
    // // for(int b = 0; b < 2; b++){
    //   for (int d = 0; d < 10; d++) {
    //     cuda_layer_11_output[b*10 + d] = layer_11_bias[d];
    //   }
    //   for (int d = 0; d < 10; d++) {
    //     for (int i = 0; i < 1; i++) {
    //       cuda_layer_11_output[b*10 + d] += 2 * __builtin_popcount((unsigned int)~(unsigned int)(layer_11_weight[d][i] ^ cuda_layer_10_output[b*1 + i])) - 32;
    //     }
    //   }
    // }
    // // end = std::chrono::high_resolution_clock::now();
    // auto l11_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());  
    // float l11_kernel_time = 0;
    // 
    // // checksum L11 
    // ofstream g11("outputs/layer11/orig.out");
    // for(int b = 0; b < 2; b++){
    //   sum_cpu = 0;
    //   for (int d = 0; d < 10; d++) {
    //     sum_cpu += cuda_layer_11_output[b*10 + d];
    //     g11<<cuda_layer_11_output[b*10 + d]<<" ";  
    //   }
    //   cout<<fixed<<"layer 11(CPU): batch "<<b<<": "<<sum_cpu<<endl;
    // }
    // cout<<endl;

    /* Layer 11 GPU */ 
    start = std::chrono::high_resolution_clock::now();
    kernel_time += layer11_gpu(cuda_layer_10_output, cuda_layer_11_output);
    end = std::chrono::high_resolution_clock::now();  
    auto l11_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
    //TODO for loop at the end? float l11_kernel_time = kernel_time-l1_kernel_time-...;
    //l11_time -= l11_kernel_time*1000000.0f; // ms->n
    float l11_kernel_time = 0;

    // // checksum L11
    // ofstream gg11("outputs/layer11/par.out");
    // for(int b = 0; b < 2; b++){
    //   sum_gpu = 0;
    //   for(int i=b*10;i<(b+1)*10;i++){
    //     sum_gpu += cuda_layer_11_output[i];
    //     gg11<<cuda_layer_11_output[i]<<" ";  
    //   }
    //   cout<<fixed<<"layer 11(GPU): batch "<<b<<": "<<sum_gpu<<endl;
    // }
    // cout<<endl;


    for(int b = 0; b < 2; b++){
      for (int i = 0; i < 10; i++) {
        pred[b*10 + i] += cuda_layer_11_output[b*10 + i];
      }
    }

    return std::make_tuple(kernel_time, l1_time, l2_time, l3_time, l4_time, l5_time, l6_time, l7_time, l8_time, l9_time, l10_time, l11_time, l1_kernel_time, l2_kernel_time, l3_kernel_time, l4_kernel_time, l5_kernel_time, l6_kernel_time, l7_kernel_time, l8_kernel_time, l9_kernel_time, l10_kernel_time, l11_kernel_time);
}

}