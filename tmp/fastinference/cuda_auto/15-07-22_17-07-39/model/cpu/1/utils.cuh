 // for cuda error checking
 #define cudaCheckErrors(msg) \
 do { \
     cudaError_t __err = cudaGetLastError(); \
     if (__err != cudaSuccess) { \
         fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
             msg, cudaGetErrorString(__err), \
             __FILE__, __LINE__); \
         fprintf(stderr, "*** FAILED - ABORTING\n"); \
         return 1; \
     } \
 } while (0)

__device__ int index3D_cuda(const int x, const int y, const int z, const int sizey, const int sizez) {
    return x*sizey*sizez + y*sizez + z;
}

__device__ int index4D_cuda(const int x, const int y, const int z, const int t, const int sizey, const int sizez, const int sizet){
    return x*sizey*sizez*sizet + y*sizez*sizet + z*sizet + t;
}

// use the second GPU on Uni-server because the first is used most of the time
void setUniGPU(){
    int devices;
    cudaGetDeviceCount(&devices);
    if(devices>1){
        cudaSetDevice(1);
    }
}