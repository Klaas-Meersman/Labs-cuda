#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

struct GPUExecutionTimes {
    float kernelTime;
    float totalTime;
};


__global__ void reverseArrayGPU(int *d_array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int swap_idx = size - 1 - idx;
    
    if (idx < size / 2) {
        int temp = d_array[idx];
        d_array[idx] = d_array[swap_idx];
        d_array[swap_idx] = temp;
    }
}

void reverseArrayCPU(int *d_array, int size) {
    for(int idx = 0; idx < size/2; idx++){
        int swap_idx = size - 1 - idx;

        int temp = d_array[idx];
        d_array[idx] = d_array[swap_idx];
        d_array[swap_idx] = temp;
    }
}


float runCPU(int array[], int size){
    //CPU
    //measuring
    const auto start = std::chrono::steady_clock::now();
    //measuring
    
    reverseArrayCPU(array, size);

    //measuring
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed_seconds{end-start};
    //measuring ;

    return elapsed_seconds.count();
}

GPUExecutionTimes runGPU(int array[], int size){
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
    reverseArrayGPU<<<blocksPerGrid, threadsPerBlock>>>(GPU_d_array, size);

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
    GPUExecutionTimes times;
    times.kernelTime = ms_kernel / 1000.0; // Convert to seconds
    times.totalTime = ms_total / 1000.0;   // Convert to seconds
    return times;
}





// Function to generate a random array
int* generateRandomArray(size_t size) {
    int* array = new int[size];
    
    std::mt19937 gen(static_cast<unsigned int>(std::time(nullptr)));
    std::uniform_int_distribution<> dis(0, 999);

    for (size_t i = 0; i < size; ++i) {
        array[i] = dis(gen);
    }

    return array;
}

int main() {
    for(int size = 10; size<10000000;size*=2){
        const int amountOfRunsPerSize = 100;
        
    
        int* h_array = generateRandomArray(size);

        float averageGPUKernelTime = 0.0f;
        float averageGPUTotalTime = 0.0f;
        float averageCPUTime = 0.0f;
        float cpuTime;


        for(int i = 0;i<amountOfRunsPerSize;i++){
            GPUExecutionTimes gpuTimes = runGPU(h_array, size);
            cpuTime = runCPU(h_array,size);

            averageGPUKernelTime += gpuTimes.kernelTime;
            averageGPUTotalTime += gpuTimes.totalTime;
            averageCPUTime += cpuTime;
        }
        averageGPUKernelTime = averageGPUKernelTime/amountOfRunsPerSize;
        averageGPUTotalTime = averageGPUTotalTime/amountOfRunsPerSize;
        averageCPUTime = averageCPUTime/amountOfRunsPerSize;


        //printf("Array size: %d\n", size);
        //printf("Elapsed time CPU: %.6e seconds\n", averageCPUTime);
        //printf("Elapsed time GPU (kernel only): %.6e seconds\n", averageGPUKernelTime);
        //printf("Elapsed time GPU (total, including data transfer): %.6e seconds\n", averageGPUTotalTime);


        printf("%d,%.6e,%.6e,%.6e\n", size, averageCPUTime, averageGPUKernelTime, averageGPUTotalTime);
        
        delete[] h_array;
    }

    return 0;
}

