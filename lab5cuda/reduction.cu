#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <algorithm>
#include <math.h>
#include <cfloat>

struct info {
    float kernelTime;
    float totalTime;
    float outcome[4];
};

__global__ void reductionMAXGPU(float *d_array, int size) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < size) ? d_array[i] : -FLT_MAX;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) d_array[blockIdx.x] = sdata[0];
}


__global__ void reductionMINGPU(float *d_array, int size) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < size) ? d_array[i] : FLT_MAX;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) d_array[blockIdx.x] = sdata[0];
}

__global__ void reductionPRODGPU(float *d_array, int size) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < size) ? d_array[i] : 1;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = sdata[tid] * sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) d_array[blockIdx.x] = sdata[0];
}

__global__ void reductionSUMGPU(float *d_array, int size) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < size) ? d_array[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) d_array[blockIdx.x] = sdata[0];
}

info runGPUReductionSync(float array_SUM[], float array_PROD[], float array_MIN[], float array_MAX[],float size){
    float *GPU_d_array_SUM;
    float *GPU_d_array_PROD;
    float *GPU_d_array_MIN;
    float *GPU_d_array_MAX;

    cudaMalloc((void**)&GPU_d_array_SUM, size * sizeof(float));
    cudaMalloc((void**)&GPU_d_array_PROD, size * sizeof(float));
    cudaMalloc((void**)&GPU_d_array_MIN, size * sizeof(float));
    cudaMalloc((void**)&GPU_d_array_MAX, size * sizeof(float));

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    int sharedMemSize = threadsPerBlock * sizeof(float);

    // Measure total time including data transfer
    cudaEvent_t start_cuda_total, stop_cuda_total;
    cudaEventCreate(&start_cuda_total);
    cudaEventCreate(&stop_cuda_total);
    cudaDeviceSynchronize();
    cudaEventRecord(start_cuda_total);

    // Measure kernel execution time separately
    cudaEvent_t start_cuda_kernel, stop_cuda_kernel;
    cudaEventCreate(&start_cuda_kernel);
    cudaEventCreate(&stop_cuda_kernel);
    cudaDeviceSynchronize();
    cudaEventRecord(start_cuda_kernel);

    cudaMemcpy(GPU_d_array_SUM, array_SUM, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    reductionSUMGPU<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(GPU_d_array_SUM, size);
    cudaDeviceSynchronize();
    cudaMemcpy(GPU_d_array_PROD, array_PROD, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    reductionPRODGPU<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(GPU_d_array_PROD, size);
    cudaDeviceSynchronize();
    cudaMemcpy(GPU_d_array_MIN, array_MIN, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    reductionMINGPU<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(GPU_d_array_MIN, size);
    cudaDeviceSynchronize();
    cudaMemcpy(GPU_d_array_MAX, array_MAX, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    reductionMAXGPU<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(GPU_d_array_MAX, size);
    cudaDeviceSynchronize();
    

    // Measure kernel execution time
    cudaDeviceSynchronize();
    cudaEventRecord(stop_cuda_kernel);
    cudaEventSynchronize(stop_cuda_kernel);
    float ms_kernel;
    cudaEventElapsedTime(&ms_kernel, start_cuda_kernel, stop_cuda_kernel);

    cudaMemcpy(array_SUM, GPU_d_array_SUM, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(array_PROD, GPU_d_array_PROD, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(array_MIN, GPU_d_array_MIN, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(array_MAX, GPU_d_array_MAX, sizeof(float), cudaMemcpyDeviceToHost);

    // Measure total time including data transfer
    cudaDeviceSynchronize();
    cudaEventRecord(stop_cuda_total);
    cudaEventSynchronize(stop_cuda_total);
    float ms_total;
    cudaEventElapsedTime(&ms_total, start_cuda_total, stop_cuda_total);

    cudaFree(GPU_d_array_SUM);
    cudaFree(GPU_d_array_PROD);
    cudaFree(GPU_d_array_MIN);
    cudaFree(GPU_d_array_MAX);

    info info;
    info.kernelTime = ms_kernel / 1000.0; 
    info.totalTime = ms_total / 1000.0;   
    info.outcome[0] = array_SUM[0];
    info.outcome[1] =  array_PROD[0];
    info.outcome[2] = array_MIN[0];
    info.outcome[3] =  array_MAX[0];
    return info;
}

info runGPUReductionAsync(float array_SUM[], float array_PROD[], float array_MIN[], float array_MAX[], float size) {
    float *GPU_d_array_SUM, *GPU_d_array_PROD, *GPU_d_array_MIN, *GPU_d_array_MAX;
    cudaStream_t stream1;
    cudaStream_t stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaMalloc((void**)&GPU_d_array_SUM, size * sizeof(float));
    cudaMalloc((void**)&GPU_d_array_PROD, size * sizeof(float));
    cudaMalloc((void**)&GPU_d_array_MIN, size * sizeof(float));
    cudaMalloc((void**)&GPU_d_array_MAX, size * sizeof(float));

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemSize = threadsPerBlock * sizeof(float);

    cudaEvent_t start_cuda_total, stop_cuda_total, start_cuda_kernel, stop_cuda_kernel;
    cudaEventCreate(&start_cuda_total);
    cudaEventCreate(&stop_cuda_total);
    cudaEventCreate(&start_cuda_kernel);
    cudaEventCreate(&stop_cuda_kernel);

    cudaEventRecord(start_cuda_total, stream1);
    cudaEventRecord(start_cuda_kernel, stream1);

    // Asynchronous memory copies and kernel launches
    cudaMemcpyAsync(GPU_d_array_SUM, array_SUM, size * sizeof(float), cudaMemcpyHostToDevice, stream1);
    reductionSUMGPU<<<blocksPerGrid, threadsPerBlock, sharedMemSize, stream2>>>(GPU_d_array_SUM, size);

    cudaMemcpyAsync(GPU_d_array_PROD, array_PROD, size * sizeof(float), cudaMemcpyHostToDevice, stream1);
    reductionPRODGPU<<<blocksPerGrid, threadsPerBlock, sharedMemSize, stream2>>>(GPU_d_array_PROD, size);

    cudaMemcpyAsync(GPU_d_array_MIN, array_MIN, size * sizeof(float), cudaMemcpyHostToDevice, stream1);
    reductionMINGPU<<<blocksPerGrid, threadsPerBlock, sharedMemSize, stream2>>>(GPU_d_array_MIN, size);

    cudaMemcpyAsync(GPU_d_array_MAX, array_MAX, size * sizeof(float), cudaMemcpyHostToDevice, stream1);
    reductionMAXGPU<<<blocksPerGrid, threadsPerBlock, sharedMemSize, stream2>>>(GPU_d_array_MAX, size);

    cudaEventRecord(stop_cuda_kernel, stream1);

    // Asynchronous copy of results back to host
    cudaMemcpyAsync(array_SUM, GPU_d_array_SUM, sizeof(float), cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(array_PROD, GPU_d_array_PROD, sizeof(float), cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(array_MIN, GPU_d_array_MIN, sizeof(float), cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(array_MAX, GPU_d_array_MAX, sizeof(float), cudaMemcpyDeviceToHost, stream1);

    cudaEventRecord(stop_cuda_total, stream1);
    cudaStreamSynchronize(stream1);

    float ms_kernel, ms_total;
    cudaEventElapsedTime(&ms_kernel, start_cuda_kernel, stop_cuda_kernel);
    cudaEventElapsedTime(&ms_total, start_cuda_total, stop_cuda_total);

    cudaFree(GPU_d_array_SUM);
    cudaFree(GPU_d_array_PROD);
    cudaFree(GPU_d_array_MIN);
    cudaFree(GPU_d_array_MAX);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    info info;
    info.kernelTime = ms_kernel / 1000.0;
    info.totalTime = ms_total / 1000.0;
    info.outcome[0] = array_SUM[0];
    info.outcome[1] = array_PROD[0];
    info.outcome[2] = array_MIN[0];
    info.outcome[3] = array_MAX[0];
    return info;
}

float* generateRandomArray(size_t size) {
    float* array = new float[size];
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f); 

    for (size_t i = 0; i < size; ++i) {
        do {
            array[i] = dis(gen);
        } while (array[i] == 0.0f || array[i] == 1.0f);
    }

    std::uniform_int_distribution<size_t> posDis(0, size - 1);
    size_t maxPosition = posDis(gen);

    array[maxPosition] = 0.999999f;
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
    const int NUM_RUNS = 20;
    const int NUM_SIZES = 11;
    const int sizes[NUM_SIZES] = {512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288};

    for (int size_index = 0; size_index < NUM_SIZES; size_index++) {
        int size = sizes[size_index];
        std::vector<float> sync_kernel_times(NUM_RUNS);
        std::vector<float> sync_total_times(NUM_RUNS);
        std::vector<float> async_kernel_times(NUM_RUNS);
        std::vector<float> async_total_times(NUM_RUNS);

        for (int run = 0; run < NUM_RUNS; run++) {
            float* temp = generateRandomArray(size);
            float* my_array = new float[size];
            memcpy(my_array, temp, size * sizeof(float));
            delete[] temp;

            // Synchronous run
            float* my_array_SUM = new float[size];
            float* my_array_PROD = new float[size];
            float* my_array_MIN = new float[size];
            float* my_array_MAX = new float[size];

            memcpy(my_array_SUM, my_array, size * sizeof(float));
            memcpy(my_array_PROD, my_array, size * sizeof(float));
            memcpy(my_array_MIN, my_array, size * sizeof(float));
            memcpy(my_array_MAX, my_array, size * sizeof(float));

            info results_sync = runGPUReductionSync(my_array_SUM, my_array_PROD, my_array_MIN, my_array_MAX, size);
            sync_kernel_times[run] = results_sync.kernelTime;
            sync_total_times[run] = results_sync.totalTime;

            // Asynchronous run
            float* my_array_SUM2 = new float[size];
            float* my_array_PROD2 = new float[size];
            float* my_array_MIN2 = new float[size];
            float* my_array_MAX2 = new float[size];

            memcpy(my_array_SUM2, my_array, size * sizeof(float));
            memcpy(my_array_PROD2, my_array, size * sizeof(float));
            memcpy(my_array_MIN2, my_array, size * sizeof(float));
            memcpy(my_array_MAX2, my_array, size * sizeof(float));

            info results_async = runGPUReductionAsync(my_array_SUM2, my_array_PROD2, my_array_MIN2, my_array_MAX2, size);
            async_kernel_times[run] = results_async.kernelTime;
            async_total_times[run] = results_async.totalTime;

            delete[] my_array;
            delete[] my_array_SUM;
            delete[] my_array_PROD;
            delete[] my_array_MIN;
            delete[] my_array_MAX;
            delete[] my_array_SUM2;
            delete[] my_array_PROD2;
            delete[] my_array_MIN2;
            delete[] my_array_MAX2;
        }
        float sync_total_median = getMedian(sync_total_times);
        float async_total_median = getMedian(async_total_times);
        printf("%d,%.6e,%.6e\n", size, sync_total_median, async_total_median);
    }
    return 0;
}