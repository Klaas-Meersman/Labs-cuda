#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <algorithm>
#include <math.h>


int maxResultSIMPLECPU = INT_MIN;
int maxResultNESTEDCPU = INT_MIN;
int maxResultATOMICGPU = INT_MIN;
int maxResultREDUCTIONGPU = INT_MIN;

struct info {
    float kernelTime;
    float totalTime;
    int MAX;
};


int simpleMAXCPU(int *d_array, int size) {
    int MAX = INT_MIN;
    for(int i = 0;i < size; i++){
        if (d_array[i] > MAX){
            MAX = d_array[i];
        }
    }
    maxResultSIMPLECPU = MAX;
    d_array[0] = MAX;
    return MAX;
}

int nestedLoopMaxCPU(int *d_array, int size){
    int step = 1;
    int interval = 1;
    while(interval < size){
        for(int i = 0; i < size-interval; i += interval*2){
            if(d_array[i+interval]>d_array[i]){
                d_array[i] = d_array[i+interval];
            }
        }
        interval *= 2;
        step++;
    }

    maxResultNESTEDCPU = d_array[0];
    return d_array[0];
}


__global__ void atomicMAXGPU(int *d_array, int size) {    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < size - 1) {  // Ensure we don't go out of bounds
        atomicMax(&d_array[0], d_array[tid]);
    }
}

__global__ void reductionMAXGPU(int *d_array, int size) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load elements into shared memory
    sdata[tid] = (i < size) ? d_array[i] : INT_MIN;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) d_array[blockIdx.x] = sdata[0];
}



info runCPUSimple(int array[], int size){
    //CPU
    //measuring
    const auto start = std::chrono::steady_clock::now();
    //measuring
    
    simpleMAXCPU(array, size);

    //measuring
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed_seconds{end-start};
    //measuring ;

    info info;
    info.kernelTime = elapsed_seconds.count(); // Convert to seconds
    info.totalTime = elapsed_seconds.count();   // Convert to seconds
    info.MAX = array[0];

    return info;
}

info runCPUNested(int array[], int size){
    //CPU
    //measuring
    const auto start = std::chrono::steady_clock::now();
    //measuring
    
    nestedLoopMaxCPU(array, size);

    //measuring
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed_seconds{end-start};
    //measuring ;

    info info;
    info.kernelTime = elapsed_seconds.count(); // Convert to seconds
    info.totalTime = elapsed_seconds.count();   // Convert to seconds
    info.MAX = array[0];

    return info;
}

info runGPUatomic(int array[], int size){
    int *GPU_d_array;
    cudaMalloc((void**)&GPU_d_array, size * sizeof(int));
    int threadsPerBlock = 1024;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Measure total time including data transfer
    cudaEvent_t start_cuda_total, stop_cuda_total;
    cudaEventCreate(&start_cuda_total);
    cudaEventCreate(&stop_cuda_total);
    cudaDeviceSynchronize();
    cudaEventRecord(start_cuda_total);

    // Copy data from host to device
    cudaMemcpy(GPU_d_array, array, size * sizeof(int), cudaMemcpyHostToDevice);

    // Measure kernel execution time separately
    cudaEvent_t start_cuda_kernel, stop_cuda_kernel;
    cudaEventCreate(&start_cuda_kernel);
    cudaEventCreate(&stop_cuda_kernel);
    cudaDeviceSynchronize();
    cudaEventRecord(start_cuda_kernel);

    // Launch kernel
    atomicMAXGPU<<<blocksPerGrid, threadsPerBlock>>>(GPU_d_array, size);

    // Measure kernel execution time
    cudaDeviceSynchronize();
    cudaEventRecord(stop_cuda_kernel);
    cudaEventSynchronize(stop_cuda_kernel);
    float ms_kernel;
    cudaEventElapsedTime(&ms_kernel, start_cuda_kernel, stop_cuda_kernel);

    // Copy result back to host
    cudaMemcpy(array, GPU_d_array, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Measure total time including data transfer
    cudaDeviceSynchronize();
    cudaEventRecord(stop_cuda_total);
    cudaEventSynchronize(stop_cuda_total);
    float ms_total;
    cudaEventElapsedTime(&ms_total, start_cuda_total, stop_cuda_total);

    // Free device memory
    cudaFree(GPU_d_array);
    info info;
    info.kernelTime = ms_kernel / 1000.0; // Convert to seconds
    info.totalTime = ms_total / 1000.0;   // Convert to seconds
    info.MAX = array[0];
    maxResultATOMICGPU = array[0];
    return info;
}

info runGPUReduction(int array[], int size) {
    int interval = 1;
    int step = 1;


    int *GPU_d_array;
    cudaMalloc((void**)&GPU_d_array, size * sizeof(int));


    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    int sharedMemSize = threadsPerBlock * sizeof(int);

    // Measure total time including data transfer
    cudaEvent_t start_cuda_total, stop_cuda_total;
    cudaEventCreate(&start_cuda_total);
    cudaEventCreate(&stop_cuda_total);
    cudaDeviceSynchronize();
    cudaEventRecord(start_cuda_total);

    // Copy data from host to device
    cudaMemcpy(GPU_d_array, array, size * sizeof(int), cudaMemcpyHostToDevice);

    // Measure kernel execution time separately
    cudaEvent_t start_cuda_kernel, stop_cuda_kernel;
    cudaEventCreate(&start_cuda_kernel);
    cudaEventCreate(&stop_cuda_kernel);
    cudaDeviceSynchronize();
    cudaEventRecord(start_cuda_kernel);

    // Launch kernel for reduction
    reductionMAXGPU<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(GPU_d_array, size);


    // Measure kernel execution time
    cudaDeviceSynchronize();
    cudaEventRecord(stop_cuda_kernel);
    cudaEventSynchronize(stop_cuda_kernel);
    float ms_kernel;
    cudaEventElapsedTime(&ms_kernel, start_cuda_kernel, stop_cuda_kernel);

    // Copy result back to host (only the first element, which contains the maximum)
    cudaMemcpy(array, GPU_d_array, sizeof(int), cudaMemcpyDeviceToHost);

    // Measure total time including data transfer
    cudaDeviceSynchronize();
    cudaEventRecord(stop_cuda_total);
    cudaEventSynchronize(stop_cuda_total);
    float ms_total;
    cudaEventElapsedTime(&ms_total, start_cuda_total, stop_cuda_total);

    // Free device memory
    cudaFree(GPU_d_array);

    info info;
    info.kernelTime = ms_kernel / 1000.0; // Convert to seconds
    info.totalTime = ms_total / 1000.0;   // Convert to seconds
    info.MAX = array[0];
    maxResultREDUCTIONGPU = array[0];
    return info;
}

//the max value is always the size here
int* generateRandomArray(size_t size) {
    int* array = new int[size];
    
    // Set up random number generation
    std::mt19937 gen(static_cast<unsigned int>(std::time(nullptr)));
    std::uniform_int_distribution<> dis(0, static_cast<int>(size) - 1);
    std::uniform_int_distribution<> valueDis(0, static_cast<int>(size) - 1);

    // Fill the array with random values
    for (size_t i = 0; i < size; ++i) {
        array[i] = valueDis(gen);
    }

    // Choose a random position for the maximum value
    size_t maxPosition = dis(gen);

    // Set the maximum value at the chosen position
    array[maxPosition] = static_cast<int>(size);

    return array;
}

void printArray(int* array, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
    printf("\n");
}


float getMedian(std::vector<float>& v) {
    size_t n = v.size() / 2;
    std::nth_element(v.begin(), v.begin()+n, v.end());
    return v[n];
}

int main() {
    const int NUM_RUNS = 80;

    for(int size = 12; size < 1000000; size *= 2) {
        std::vector<float> timesCPUSimple(NUM_RUNS);
        std::vector<float> timesCPUNested(NUM_RUNS);
        std::vector<float> timesGPUAtomic(NUM_RUNS);
        std::vector<float> timesGPUReduction(NUM_RUNS);

        for (int run = 0; run < NUM_RUNS; run++) {
            int* h_array1 = generateRandomArray(size);
            int* h_array2 = generateRandomArray(size);
            int* h_array3 = generateRandomArray(size);
            int* h_array4 = generateRandomArray(size);

            cudaDeviceSynchronize();
            timesCPUSimple[run] = runCPUSimple(h_array1, size).kernelTime;
            cudaDeviceSynchronize();
            timesCPUNested[run] = runCPUNested(h_array2, size).kernelTime;
            cudaDeviceSynchronize();
            timesGPUAtomic[run] = runGPUatomic(h_array3, size).kernelTime;
            cudaDeviceSynchronize();
            timesGPUReduction[run] = runGPUReduction(h_array4, size).kernelTime;
            cudaDeviceSynchronize();

            delete[] h_array1;
            delete[] h_array2;
            delete[] h_array3;
            delete[] h_array4;
        }

        float medianCPUSimple = getMedian(timesCPUSimple);
        float medianCPUNested = getMedian(timesCPUNested);
        float medianGPUAtomic = getMedian(timesGPUAtomic);
        float medianGPUReduction = getMedian(timesGPUReduction);

        printf("%d,%.6e,%.6e,%.6e,%.6e\n", size,
               medianCPUSimple,
               medianCPUNested,
               medianGPUAtomic,
               medianGPUReduction);
    }

    return 0;
}


