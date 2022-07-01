#include <iostream>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <cmath>
#include <assert.h>

#include "model.h"

namespace FAST_INFERENCE {}
using namespace FAST_INFERENCE;

// unsigned int BATCH_SIZE = 2;

auto read_csv(std::string &path) {
	std::vector<std::vector<FEATURE_TYPE>> X;
	std::vector<unsigned int> Y;

	std::ifstream file(path);
	// if (!file_exists(path)) {
	// 	throw std::runtime_error("File not found " + path);
	// }
	std::string header;
	std::getline(file, header);

	unsigned int label_pos = 0;
	std::stringstream ss(header);
	std::string entry;
	while (std::getline(ss, entry, ',')) {
		if (entry == "label") {
			break;
		} else {
			label_pos++;
		}
	}
	std::cout << "label_pos: " << label_pos << std::endl;

	if (file.is_open()) {
		std::string line;
		while (std::getline(file, line)) {
			if (line.size() > 0) {
				std::stringstream ss(line);
				entry = "";

				unsigned int i = 0;
				std::vector<FEATURE_TYPE> x;
				while (std::getline(ss, entry, ',')) {
					if (i == label_pos) {
						Y.push_back(static_cast<unsigned int>(std::stoi(entry)));
					} else {
						x.push_back(static_cast<FEATURE_TYPE>(std::stof(entry)));
					}
					++i;
				}
				X.push_back(x);
			}
		}
		file.close();
	}
	return std::make_tuple(X,Y);
}

auto benchmark(std::vector<std::vector<FEATURE_TYPE>> &X, std::vector<unsigned int> &Y, unsigned int repeat) {

	LABEL_TYPE * output = new LABEL_TYPE[N_CLASSES*BATCH_SIZE];
    unsigned int n_features = X[0].size();
	unsigned int matches = 0;

	size_t xsize = X.size();
	// size_t xsize = 2; // for testing;

	const unsigned int imgsize = X[0].size();
	
	std::cout<<"Dataset size: "<<xsize<<std::endl;
	std::cout<<"Image size: "<<imgsize<<std::endl;
	std::cout<<"Batch size: "<<BATCH_SIZE<<std::endl;
	std::cout<<"=> Executing "<<xsize<<" images in "<<ceil(float(xsize)/BATCH_SIZE)<<" batches of "<<BATCH_SIZE<<"..."<<std::endl;
	std::cout<<std::endl;

	float total_kernel_time = 0;
	// TODO generate these:
    float l1_time = 0, l2_time = 0, l3_time = 0, l4_time = 0, l5_time = 0, l6_time = 0, l7_time = 0, l8_time = 0, l9_time = 0, l10_time = 0, l11_time = 0;
    float l1_kernel_time = 0, l2_kernel_time = 0, l3_kernel_time = 0, l4_kernel_time = 0, l5_kernel_time = 0, l6_kernel_time = 0, l7_kernel_time = 0, l8_kernel_time = 0, l9_kernel_time = 0, l10_kernel_time = 0, l11_kernel_time = 0;

    auto start = std::chrono::high_resolution_clock::now();
    for (unsigned int k = 0; k < repeat; ++k) {
    	matches = 0;
	    /* using ceil() makes sure to execute even when division is not uniform: */
		for (unsigned int b = 0; b < ceil(float(xsize)/BATCH_SIZE); b++){
	        std::fill(output, output+N_CLASSES*BATCH_SIZE, 0);
			// TODO make label as array for multiple batches
	        unsigned int label[BATCH_SIZE];

	        // Note: To make this code more universially applicable we define predict to be the correct function
	        //       which is given in the command line argument. For example, a RidgeClassifier is compiled with
	        //          cmake . -DMODEL_NAME=RidgeClassifier
	        //       where in the cmake file we define
	        //          SET(MODELNAME "" CACHE STRING "Name of the model / classifier. Usually found in the corresponding JSON file.")
	        //          target_compile_definitions(testCode PRIVATE -Dpredict=predict_${MODELNAME})
			//int const * const x = &X[i*NUM_FEATURES];

			// FEATURE_TYPE const * const x = &X[i][0];
			FEATURE_TYPE x[BATCH_SIZE][imgsize];

			size_t BSIZE = (b == xsize/BATCH_SIZE) ? (xsize % BATCH_SIZE) : BATCH_SIZE;
			for(size_t i=0; i<BSIZE; i++){
				for(size_t n=0; n<X[b*BSIZE + i].size(); n++){ // imgsize
					x[i][n] = X[b*BSIZE + i][n];
				}
				// x[i][0][0] = X[b*BSIZE + i][0];
				label[i] = Y[b*BSIZE + i];
			}
			
			// TODO: generate these:
			float kernel_time, l1t, l2t, l3t, l4t, l5t, l6t, l7t, l8t, l9t, l10t, l11t, l1kt, l2kt, l3kt, l4kt, l5kt, l6kt, l7kt, l8kt, l9kt, l10kt, l11kt;
			std::tie(kernel_time, l1t, l2t, l3t, l4t, l5t, l6t, l7t, l8t, l9t, l10t, l11t, l1kt, l2kt, l3kt, l4kt, l5kt, l6kt, l7kt, l8kt, l9kt, l10kt, l11kt) = predict(&x[0][0], output);
			total_kernel_time += kernel_time;
			l1_time += l1t, l2_time += l2t, l3_time += l3t, l4_time += l4t, l5_time += l5t, l6_time += l6t, l7_time += l7t, l8_time += l8t, l9_time += l9t, l10_time += l10t, l11_time += l11t;
			l1_kernel_time += l1kt, l2_kernel_time += l2kt, l3_kernel_time += l3kt, l4_kernel_time += l4kt, l5_kernel_time += l5kt, l6_kernel_time += l6kt, l7_kernel_time += l7kt, l8_kernel_time += l8kt, l9_kernel_time += l9kt, l10_kernel_time += l10kt, l11_kernel_time += l11kt;

			// for(int b = 0; b<BSIZE; b++){
			// 	for (int i=0;i<10;i++){
			// 		std::cout<<output[b*10+i]<<" ";
			// 	}
			// 	std::cout<<std::endl;
			// }
			// std::cout<<std::endl;

			// TODO adapt for multiple batches in the matches code:
			if constexpr (N_CLASSES >= 2) {
				for(unsigned int b = 0; b < BSIZE; b++){
					LABEL_TYPE max = output[b*N_CLASSES];
					unsigned int argmax = 0;
					for (unsigned int j = 1; j < N_CLASSES; j++) {
						if (output[b*N_CLASSES + j] > max) {
							max = output[b*N_CLASSES + j];
							argmax = j;
						}
					}
					// std::cout<<"ximage: "<<b<<" | "<<"label: "<<label[b]<<", argmax: "<<argmax<<std::endl;
					if (argmax == label[b]) {
						// std::cout<<"image: "<<b<<" | "<<"label: "<<label[b]<<", argmax: "<<argmax<<std::endl;
						++matches;
					}
				}
			} else {
				if ( (output[0] < 0 && label[b] == 0) || (output[0] >= 0 && label[b] == 1) ) {
					++matches;
				}
			} 
	    }
    }

	delete[] output;

    auto end = std::chrono::high_resolution_clock::now();   
	// TOTAL TIMES AVERAGED BY DIVIDING WITH repeat
    auto total_cpu_time = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()) / (repeat);
	total_kernel_time /= repeat;
	total_cpu_time -= total_kernel_time;
	auto cpu_time = static_cast<float>(total_cpu_time) / xsize;
	auto kernel_time = static_cast<float>(total_kernel_time) / xsize;

	float accuracy = static_cast<float>(matches) / X.size() * 100.f;
    printf("Matches: %d/10000\n", matches);

	// TODO generate these:
	return std::make_tuple(accuracy, total_cpu_time, cpu_time, total_kernel_time, kernel_time,
		l1_time, l2_time, l3_time, l4_time, l5_time, l6_time, l7_time, l8_time, l9_time, l10_time, l11_time,
        l1_kernel_time, l2_kernel_time, l3_kernel_time, l4_kernel_time, l5_kernel_time, l6_kernel_time, l7_kernel_time, l8_kernel_time, l9_kernel_time, l10_kernel_time, l11_kernel_time);
}

int main (int argc, char *argv[]) {
	if (argc <= 2) {
		std::cout << "Please provide two arguments: path n_repetitions" << std::endl;
	}
	std::string path = std::string(argv[1]);
	unsigned int repeat = std::stoi(argv[2]);

	auto data = read_csv(path);

	assert(std::get<0>(data).size() > 0);
	assert(std::get<0>(data).size() == std::get<1>(data).size());
	assert(std::get<0>(data)[0].size() == N_FEATURES);

    std::cout << std::endl << "RUNNING BENCHMARK WITH " << repeat << " REPETITIONS" << std::endl << std::endl;
    auto results = benchmark(std::get<0>(data), std::get<1>(data), repeat);
    
    std::cout << "Accuracy: " << std::get<0>(results) << " %" << std::endl;
    // std::cout << "Latency: " << std::get<1>(results) << " [ms/elem]" << std::endl;
	#ifdef REF_ACCURACY
		float difference = std::get<0>(results) - REF_ACCURACY;
		std::cout << "Reference Accuracy: " << REF_ACCURACY << " %" << std::endl << std::endl;
		std::cout << "Difference: " << difference << std::endl;
	    
        std::cout << std::get<0>(results) << "," << REF_ACCURACY << "," << difference << "," << std::get<1>(results) << std::endl;
	#else
        std::cout << std::get<0>(results) << "," << "," << "," << std::get<1>(results) << std::endl;
    #endif

	if(repeat > 1){
		std::cout<<std::endl<<"ALGORITHM REPEATED " << repeat << " > 1 TIMES => TOTAL TIMES ARE AVERAGED (DIVIDED BY " << repeat << std::endl;
	}
	printf("\n");
    printf("Total CPU time: %.2f [s] => Latency: %.4f [ms/elem]\n", std::get<1>(results)/1000.0f, std::get<2>(results));
    printf("Total GPU time: %.2f [s] => Latency: %.4f [ms/elem]\n", std::get<3>(results)/1000.0f, std::get<4>(results));
    printf("\n");

	// for profiling layers
	// TODO generate these:
    float l1_time = std::get<5>(results)/1000000000.0f; // ns / 1e9 -> s
    float l2_time = std::get<6>(results)/1000000000.0f; // ns / 1e9 -> s
    float l3_time = std::get<7>(results)/1000000000.0f; // ns / 1e9 -> s
    float l4_time = std::get<8>(results)/1000000000.0f; // ns / 1e9 -> s
    float l5_time = std::get<9>(results)/1000000000.0f; // ns / 1e9 -> s
    float l6_time = std::get<10>(results)/1000000000.0f; // ns / 1e9 -> s
    float l7_time = std::get<11>(results)/1000000000.0f; // ns / 1e9 -> s
    float l8_time = std::get<12>(results)/1000000000.0f; // ns / 1e9 -> s
    float l9_time = std::get<13>(results)/1000000000.0f; // ns / 1e9 -> s
    float l10_time = std::get<14>(results)/1000000000.0f; // ns / 1e9 -> s
    float l11_time = std::get<15>(results)/1000000000.0f; // ns / 1e9 -> s

	float l1_ktime = std::get<16>(results)/1000.0f; // ms / 1e3 -> s
	float l2_ktime = std::get<17>(results)/1000.0f; // ms / 1e3 -> s
	float l3_ktime = std::get<18>(results)/1000.0f; // ms / 1e3 -> s
	float l4_ktime = std::get<19>(results)/1000.0f; // ms / 1e3 -> s
	float l5_ktime = std::get<20>(results)/1000.0f; // ms / 1e3 -> s
	float l6_ktime = std::get<21>(results)/1000.0f; // ms / 1e3 -> s
	float l7_ktime = std::get<22>(results)/1000.0f; // ms / 1e3 -> s
	float l8_ktime = std::get<23>(results)/1000.0f; // ms / 1e3 -> s
	float l9_ktime = std::get<24>(results)/1000.0f; // ms / 1e3 -> s
	float l10_ktime = std::get<25>(results)/1000.0f; // ms / 1e3 -> s
	float l11_ktime = std::get<26>(results)/1000.0f; // ms / 1e3 -> s

    float sum_l = l1_time + l2_time + l3_time + l4_time + l5_time + l6_time + l7_time + l8_time + l9_time + l10_time + l11_time;
    float sum_kl = l1_ktime + l2_ktime + l3_ktime + l4_ktime + l5_ktime + l6_ktime + l7_ktime + l8_ktime + l9_ktime + l10_ktime + l11_ktime;

	printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 1 time:", l1_time, "Ratio:", (l1_time/sum_l)*100, "kernel:", l1_ktime, "kRatio:", (l1_ktime/sum_kl)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 2 time:", l2_time, "Ratio:", (l2_time/sum_l)*100, "kernel:", l2_ktime, "kRatio:", (l2_ktime/sum_kl)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 3 time:", l3_time, "Ratio:", (l3_time/sum_l)*100, "kernel:", l3_ktime, "kRatio:", (l3_ktime/sum_kl)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 4 time:", l4_time, "Ratio:", (l4_time/sum_l)*100, "kernel:", l4_ktime, "kRatio:", (l4_ktime/sum_kl)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 5 time:", l5_time, "Ratio:", (l5_time/sum_l)*100, "kernel:", l5_ktime, "kRatio:", (l5_ktime/sum_kl)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 6 time:", l6_time, "Ratio:", (l6_time/sum_l)*100, "kernel:", l6_ktime, "kRatio:", (l6_ktime/sum_kl)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 7 time:", l7_time, "Ratio:", (l7_time/sum_l)*100, "kernel:", l7_ktime, "kRatio:", (l7_ktime/sum_kl)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 8 time:", l8_time, "Ratio:", (l8_time/sum_l)*100, "kernel:", l8_ktime, "kRatio:", (l8_ktime/sum_kl)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 9 time:", l9_time, "Ratio:", (l9_time/sum_l)*100, "kernel:", l9_ktime, "kRatio:", (l9_ktime/sum_kl)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 10 time:", l10_time, "Ratio:", (l10_time/sum_l)*100, "kernel:", l10_ktime, "kRatio:", (l10_ktime/sum_kl)*100);
    printf("%-15s %-10.2f [s], %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer 11 time:", l11_time, "Ratio:", (l11_time/sum_l)*100, "kernel:", l11_ktime, "kRatio:", (l11_ktime/sum_kl)*100);
    printf("\n");
    printf("%-15s %.2f [s]\n%-15s %.2f [s]\n", "Total time:", sum_l, "Total ktime:", sum_kl);

    return 0;
}