#include <iostream>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <cmath>
#include <assert.h>
#include <experimental/filesystem>
#include <iomanip>

#include "model.h"

namespace FAST_INFERENCE {}
using namespace FAST_INFERENCE;

// unsigned int BATCH_SIZE = 2;
// const int NR_LAYERS = 11; // TODO add in Cmake

auto read_csv(std::string &path) {
	std::vector<std::vector<FEATURE_TYPE>> X;
	std::vector<unsigned int> Y;

	std::ifstream file(path);
	// if (!file_exists(path)) {
	// 	throw std::runtime_error("File not found " + path);
	// }
	std::string header;
	std::getline(file, header);
	std::cout<<path<<std::endl;

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

auto benchmark(std::vector<std::vector<FEATURE_TYPE>> &X, std::vector<unsigned int> &Y, unsigned int repeat, std::string prof_path, float * ln_total, float * ln_kernel_total) {

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

	std::ofstream f(prof_path, std::ios::app);  // append to file to not overwrite here
	f<<"Dataset size: "<<xsize<<std::endl;
	f<<"Image size: "<<imgsize<<std::endl;
	f<<"Batch size: "<<BATCH_SIZE<<std::endl;
	f<<"=> Executing "<<xsize<<" images in "<<ceil(float(xsize)/BATCH_SIZE)<<" batches of "<<BATCH_SIZE<<"..."<<std::endl;
	f<<std::endl;

	float total_kernel_time = 0;
	
	float * ln_times = new float[2*NR_LAYERS+1];
	std::fill(ln_times, ln_times+(2*NR_LAYERS+1), 0);

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
			//std::cout<<b<<" "<<BSIZE<<" "<<std::endl;
			for(size_t i=0; i<BSIZE; i++){
				for(size_t n=0; n<X[b*BATCH_SIZE + i].size(); n++){ // imgsize
					x[i][n] = X[b*BATCH_SIZE + i][n];
				}
				label[i] = Y[b*BATCH_SIZE + i];
			}
			
			predict_model(&x[0][0], output, ln_times);
			
			// first value is the total kernel time
			total_kernel_time += ln_times[0];
			
			// next NR_LAYERS values are CPU times for individual layers
			int j = 0;
			for(int i = 1; i < NR_LAYERS+1; i++){
				ln_total[j++] += ln_times[i];
			}

			// next NR_LAYERS values are GPU kernel times for individual layers
			j = 0;
			for(int i = NR_LAYERS+1; i < 2*NR_LAYERS+1; i++){
				ln_kernel_total[j++] += ln_times[i];
			}

			// for(int b = 0; b<BSIZE; b++){
			// 	for (int i=0;i<10;i++){
			// 		std::cout<<output[b*10+i]<<" ";
			// 	}
			// 	std::cout<<std::endl;
			// }
			// std::cout<<std::endl;

			if constexpr (N_CLASSES >= 2) {
				for(unsigned int i = 0; i < BSIZE; i++){
					LABEL_TYPE max = output[i*N_CLASSES];
					unsigned int argmax = 0;
					for (unsigned int j = 1; j < N_CLASSES; j++) {
						if (output[i*N_CLASSES + j] > max) {
							max = output[i*N_CLASSES + j];
							argmax = j;
						}
					}
					//std::cout<<"ximage: "<<b*BATCH_SIZE+i<<" | "<<"label: "<<label[i]<<", argmax: "<<argmax<<std::endl;
					if (argmax == label[i]) {
						// std::cout<<"image: "<<b*BATCH_SIZE+i<<" | "<<"label: "<<label[b]<<", argmax: "<<argmax<<std::endl;
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

	//printf("\n");
    printf("Matches: %d/10000\n", matches);
	f << "Matches: " << matches << "/10000\n";

	f.close();

	return std::make_tuple(accuracy, total_cpu_time, cpu_time, total_kernel_time, kernel_time);
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

	if(!std::filesystem::exists(std::string("profiles/") + IMPL)){
		std::filesystem::create_directories(std::string("profiles/") + IMPL);
	}
	std::string profile_path = std::string("profiles/") + IMPL + "/" + "timings_" + IMPL + "_" + std::to_string(BATCH_SIZE) + ".out";
	std::string profile_path_auto = std::string(OUT_PATH) + "/" + "timings_" + IMPL + "_" + std::to_string(BATCH_SIZE) + ".out";

	//profile_path = profile_path_auto;

	std::ofstream fout(profile_path); // no append when running benchmark again

	std::cout << std::endl << "Using CUDA profile " << IMPL << std::endl;
	fout << "Using CUDA profile " << IMPL << std::endl << std::endl;

    std::cout << std::endl << "RUNNING BENCHMARK WITH " << repeat << " REPETITIONS" << std::endl << std::endl;
	fout << "RUNNING BENCHMARK WITH " << repeat << " REPETITIONS" << std::endl << std::endl;

	fout.close();

	float * ln_total = new float[NR_LAYERS];
	float * ln_kernel_total = new float[NR_LAYERS];
	std::fill(ln_total, ln_total+NR_LAYERS, 0);
	std::fill(ln_kernel_total, ln_kernel_total+NR_LAYERS, 0);

    auto results = benchmark(std::get<0>(data), std::get<1>(data), repeat, profile_path, ln_total, ln_kernel_total);
    
	std::ofstream fo(profile_path, std::ios::app); // append to file to not overwrite here

    std::cout << "Accuracy: " << std::get<0>(results) << " %" << std::endl;
	fo << "Accuracy: " << std::get<0>(results) << " %" << std::endl;
    // std::cout << "Latency: " << std::get<1>(results) << " [ms/elem]" << std::endl;
	#ifdef REF_ACCURACY
		float difference = std::get<0>(results) - REF_ACCURACY;
		std::cout << "Reference Accuracy: " << REF_ACCURACY << " %" << std::endl << std::endl;
		std::cout << "Difference: " << difference << std::endl;
        std::cout << std::get<0>(results) << "," << REF_ACCURACY << "," << difference << "," << std::get<1>(results) << std::endl;

		fo << "Reference Accuracy: " << REF_ACCURACY << " %" << std::endl << std::endl;
		fo << "Difference: " << difference << std::endl;
        fo << std::get<0>(results) << "," << REF_ACCURACY << "," << difference << "," << std::get<1>(results) << std::endl;
	#else
        std::cout << std::get<0>(results) << "," << "," << "," << std::get<1>(results) << std::endl;
		fo << std::get<0>(results) << "," << "," << "," << std::get<1>(results) << std::endl;
    #endif

	if(repeat > 1){
		std::cout<<std::endl<<"ALGORITHM REPEATED " << repeat << " > 1 TIMES => TOTAL TIMES ARE AVERAGED (DIVIDED BY " << repeat << ")" << std::endl;
		fo<<std::endl<<"ALGORITHM REPEATED " << repeat << " > 1 TIMES => TOTAL TIMES ARE AVERAGED (DIVIDED BY " << repeat << ")" << std::endl;
	}
	printf("\n");
    printf("Total CPU time: %.2f [s] => Latency: %.4f [ms/elem]\n", std::get<1>(results)/1000.0f, std::get<2>(results));
    printf("Total GPU time: %.2f [s] => Latency: %.4f [ms/elem]\n", std::get<3>(results)/1000.0f, std::get<4>(results));
    printf("\n");

	fo << "\n";
	fo << std::fixed << "Total CPU time: " << std::setprecision(2) << std::get<1>(results)/1000.0f << " [s] => Latency: " << std::get<2>(results) << " [ms/elem]\n";
	fo << std::fixed << "Total GPU time: " << std::setprecision(2) << std::get<3>(results)/1000.0f << " [s] => Latency: " << std::get<4>(results) << " [ms/elem]\n";
	fo << "\n";

	// for(int j = 0; j < NR_LAYERS; j++){
	// 	std::cout<<j<<": "<<ln_total[j]<<std::endl;
	// }
	// std::cout<<std::endl;

	// for(int j = 0; j < NR_LAYERS; j++){
	// 	std::cout<<j<<": "<<ln_kernel_total[j]<<std::endl;
	// }
	// std::cout<<std::endl;

	// for profiling layers
	float sum_l = 0, sum_kl = 0;
	for (int j = 0; j < NR_LAYERS; j++){
		ln_total[j] /= 1000000000.0f; // ns / 1e9 -> s
		ln_kernel_total[j] /= 1000.0f; // ms / 1e3 -> s
		
		sum_l += ln_total[j];
		sum_kl += ln_kernel_total[j];
	}

	for(int j = 0; j < NR_LAYERS; j++){
		// printf("%s%-2d%-15s %-10.2f [s] %-10s %-5.2f% => %-5s %-5.2f [s] %-10s %-5.2f%\n", "Layer ", j+1," time:", ln_total[j], "Ratio:", (ln_total[j]/sum_l)*100, "kernel:", ln_kernel_total[j], "kRatio:", (ln_kernel_total[j]/sum_kl)*100);
		printf("%s%d%s %.2f [s] %s %.2f % => %s %.2f [s] %s %.2f %\n", "Layer ", j+1," time:", ln_total[j], "Ratio:", (ln_total[j]/sum_l)*100, "kernel:", ln_kernel_total[j], "kRatio:", (ln_kernel_total[j]/sum_kl)*100);
		fo << std::fixed << std::setw(7) << "Layer "<< std::setw(2) << j <<" time: " << std::setw(10) << std::setprecision(2) << ln_total[j] << std::setw(10) << "[s], Ratio: " << std::setw(7) << std::setprecision(2) << (ln_total[j]/sum_l)*100 << std::setw(10) << "% => kernel: " << std::setw(10) << std::setprecision(2) << ln_kernel_total[j] << std::setw(10) << "[s], kRatio: " << std::setw(7) << std::setprecision(2) << (ln_kernel_total[j]/sum_kl)*100 << "%\n";
	}
	printf("\n");
    printf("%-15s %.2f [s]\n", "Total time:", sum_l + sum_kl);
	fo << "\n";
	fo << std::fixed << "Total time: " << std::setprecision(2) << sum_l + sum_kl << "s\n";

    return 0;
}
