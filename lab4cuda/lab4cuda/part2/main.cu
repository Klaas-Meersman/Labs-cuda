/*
 * Code snippet for importing / exporting image data.
 *
 * To convert an image to a pixel map, run `convert <name>.<extension> <name>.ppm
 *
 */
#include <cstdint>  // Data types
#include <iostream> // File operations
#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <algorithm>
#include <math.h>

struct info
{
    float kernelTime;
    float totalTime;
};

__global__ void matrixMulKernel(int *A, int *B, int *C, int rowsA, int colsA, int colsB)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB)
    {
        int sum = 0;
        for (int k = 0; k < colsA; ++k)
        {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}


info matrix_multiplication_run(int rowsA, int colsA, int rowsB, int colsB)
{
    // Define small matrices A (FxG), B (GxH), and C (FxH)

    if (colsA != rowsB)
    {
        printf("Matrix dimensions are not compatible for multiplication\n");
        return {0, 0}; // Return early if dimensions are invalid
    }

    const int F = rowsA;
    const int G = colsA;
    const int H = colsB;

    // Initialize matrices as vectors
    std::vector<int> h_A(F * G);
    std::vector<int> h_B(G * H);
    std::vector<int> h_C(F * H);

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distribution(0, 20);

    // Fill matrices A and B with random values
    for (int i = 0; i < F; ++i)
    {
        for (int j = 0; j < G; ++j)
        {
            h_A[i * G + j] = distribution(gen);
        }
    }
    for (int i = 0; i < G; ++i)
    {
        for (int j = 0; j < H; ++j)
        {
            h_B[i * H + j] = distribution(gen);
        }
    }

/*     // Print matrix A
    printf("Matrix A:\n");
    for (int i = 0; i < F; ++i)
    {
        for (int j = 0; j < G; ++j)
        {
            printf("%d\t", h_A[i * G + j]);
        }
        printf("\n");
    }

    // Print matrix B
    printf("Matrix B:\n");
    for (int i = 0; i < G; ++i)
    {
        for (int j = 0; j < H; ++j)
        {
            printf("%d\t", h_B[i * H + j]);
        }
        printf("\n");
    } */

    // Configure grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (H + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (F + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Allocate device memory
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, F * G * sizeof(int));
    cudaMalloc(&d_B, G * H * sizeof(int));
    cudaMalloc(&d_C, F * H * sizeof(int));

    // Measure total time including data transfer
    cudaEvent_t start_cuda_total, stop_cuda_total;
    cudaEventCreate(&start_cuda_total);
    cudaEventCreate(&stop_cuda_total);
    cudaDeviceSynchronize();
    cudaEventRecord(start_cuda_total);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A.data(), F * G * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), G * H * sizeof(int), cudaMemcpyHostToDevice);

    // Measure kernel execution time separately
    cudaEvent_t start_cuda_kernel, stop_cuda_kernel;
    cudaEventCreate(&start_cuda_kernel);
    cudaEventCreate(&stop_cuda_kernel);
    cudaDeviceSynchronize();
    cudaEventRecord(start_cuda_kernel);

    // Launch the kernel
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, F, G, H);

    // Measure kernel execution time
    cudaDeviceSynchronize();
    cudaEventRecord(stop_cuda_kernel);
    cudaEventSynchronize(stop_cuda_kernel);
    float ms_kernel;
    cudaEventElapsedTime(&ms_kernel, start_cuda_kernel, stop_cuda_kernel);

    // Copy the result back to the host
    cudaMemcpy(h_C.data(), d_C, F * H * sizeof(int), cudaMemcpyDeviceToHost);

    // Measure total time including data transfer
    cudaDeviceSynchronize();
    cudaEventRecord(stop_cuda_total);
    cudaEventSynchronize(stop_cuda_total);
    float ms_total;
    cudaEventElapsedTime(&ms_total, start_cuda_total, stop_cuda_total);

    // Print the result
/*     printf("Matrix C (Result of A x B):\n");
    for (int i = 0; i < F; ++i)
    {
        for (int j = 0; j < H; ++j)
        {
            printf("%d\t", h_C[i * H + j]);
        }
        printf("\n");
    } */

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    info info;
    info.kernelTime = ms_kernel / 1000.0; // Convert to seconds
    info.totalTime = ms_total / 1000.0;   // Convert to seconds
    return info;
}



__global__ void matrixMulSharedKernel(int *A, int *B, int *C, int rowsA, int colsA, int colsB)
{
    // Define shared memory tiles
    __shared__ int sharedA[16][16];
    __shared__ int sharedB[16][16];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;

    // Loop over tiles of the matrices
    for (int tile = 0; tile < (colsA + 15) / 16; ++tile)
    {
        // Load elements into shared memory
        if (row < rowsA && (tile * 16 + threadIdx.x) < colsA)
            sharedA[threadIdx.y][threadIdx.x] = A[row * colsA + tile * 16 + threadIdx.x];
        else
            sharedA[threadIdx.y][threadIdx.x] = 0;

        if (col < colsB && (tile * 16 + threadIdx.y) < colsA)
            sharedB[threadIdx.y][threadIdx.x] = B[(tile * 16 + threadIdx.y) * colsB + col];
        else
            sharedB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads(); // Synchronize threads to ensure shared memory is loaded

        // Compute partial sum for the tile
        for (int k = 0; k < 16; ++k)
        {
            sum += sharedA[threadIdx.y][k] * sharedB[k][threadIdx.x];
        }

        __syncthreads(); // Synchronize before loading the next tile
    }

    // Write the result to the output matrix
    if (row < rowsA && col < colsB)
    {
        C[row * colsB + col] = sum;
    }
}


info matrix_multiplication_run_shared(int rowsA, int colsA, int rowsB, int colsB)
{
    // Define small matrices A (FxG), B (GxH), and C (FxH)
    if (colsA != rowsB)
    {
        printf("Matrix dimensions are not compatible for multiplication\n");
        return {0, 0}; // Return early if dimensions are invalid
    }

    const int F = rowsA;
    const int G = colsA;
    const int H = colsB;

    // Initialize matrices as vectors
    std::vector<int> h_A(F * G);
    std::vector<int> h_B(G * H);
    std::vector<int> h_C(F * H);

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distribution(0, 20);

    // Fill matrices A and B with random values
    for (int i = 0; i < F; ++i)
    {
        for (int j = 0; j < G; ++j)
        {
            h_A[i * G + j] = distribution(gen);
        }
    }
    for (int i = 0; i < G; ++i)
    {
        for (int j = 0; j < H; ++j)
        {
            h_B[i * H + j] = distribution(gen);
        }
    }

/*     // Print matrix A
    printf("Matrix A:\n");
    for (int i = 0; i < F; ++i)
    {
        for (int j = 0; j < G; ++j)
        {
            printf("%d\t", h_A[i * G + j]);
        }
        printf("\n");
    }

    // Print matrix B
    printf("Matrix B:\n");
    for (int i = 0; i < G; ++i)
    {
        for (int j = 0; j < H; ++j)
        {
            printf("%d\t", h_B[i * H + j]);
        }
        printf("\n");
    } */

    // Configure grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (H + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (F + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaFuncSetCacheConfig(matrixMulSharedKernel, cudaFuncCachePreferShared);

    // Allocate device memory
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, F * G * sizeof(int));
    cudaMalloc(&d_B, G * H * sizeof(int));
    cudaMalloc(&d_C, F * H * sizeof(int));

    // Measure total time including data transfer
    cudaEvent_t start_cuda_total, stop_cuda_total;
    cudaEventCreate(&start_cuda_total);
    cudaEventCreate(&stop_cuda_total);
    cudaDeviceSynchronize();
    cudaEventRecord(start_cuda_total);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A.data(), F * G * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), G * H * sizeof(int), cudaMemcpyHostToDevice);

    // Measure kernel execution time separately
    cudaEvent_t start_cuda_kernel, stop_cuda_kernel;
    cudaEventCreate(&start_cuda_kernel);
    cudaEventCreate(&stop_cuda_kernel);
    cudaDeviceSynchronize();
    cudaEventRecord(start_cuda_kernel);

    // Launch the shared memory kernel
    matrixMulSharedKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, F, G, H);

    // Measure kernel execution time
    cudaDeviceSynchronize();
    cudaEventRecord(stop_cuda_kernel);
    cudaEventSynchronize(stop_cuda_kernel);
    float ms_kernel;
    cudaEventElapsedTime(&ms_kernel, start_cuda_kernel, stop_cuda_kernel);

    // Copy the result back to the host
    cudaMemcpy(h_C.data(), d_C, F * H * sizeof(int), cudaMemcpyDeviceToHost);

    // Measure total time including data transfer
    cudaDeviceSynchronize();
    cudaEventRecord(stop_cuda_total);
    cudaEventSynchronize(stop_cuda_total);
    float ms_total;
    cudaEventElapsedTime(&ms_total, start_cuda_total, stop_cuda_total);

/*     // Print the result
    printf("Matrix C (Result of A x B using Shared Memory Kernel):\n");
    for (int i = 0; i < F; ++i)
    {
        for (int j = 0; j < H; ++j)
        {
            printf("%d\t", h_C[i * H + j]);
        }
        printf("\n");
    } */

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    info info;
    info.kernelTime = ms_kernel / 1000.0; // Convert to seconds
    info.totalTime = ms_total / 1000.0;   // Convert to seconds
    return info;
}


// Constant memory matrix
__constant__ int const_B[64 * 64]; // Example: Assuming max matrix size is 256*256
__constant__ int const_A[64 * 64];

__global__ void matrixMulConstMemoryKernel(int *A, int *B, int *C, int rowsA, int colsA, int colsB)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB)
    {
        int sum = 0;
        for (int k = 0; k < colsA; ++k)
        {
            sum += const_A[row * colsA + k] * const_B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

info matrix_multiplication_run_const(int rowsA, int colsA, int rowsB, int colsB)
{
    // Define small matrices A (FxG), B (GxH), and C (FxH)

    if (colsA != rowsB)
    {
        printf("Matrix dimensions are not compatible for multiplication\n");
        return {0, 0}; // Return early if dimensions are invalid
    }

    const int F = rowsA;
    const int G = colsA;
    const int H = colsB;

    // Initialize matrices as vectors
    std::vector<int> h_A(F * G);
    std::vector<int> h_B(G * H);
    std::vector<int> h_C(F * H);

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distribution(0, 20);

    // Fill matrices A and B with random values
    for (int i = 0; i < F; ++i)
    {
        for (int j = 0; j < G; ++j)
        {
            h_A[i * G + j] = distribution(gen);
        }
    }
    for (int i = 0; i < G; ++i)
    {
        for (int j = 0; j < H; ++j)
        {
            h_B[i * H + j] = distribution(gen);
        }
    }

    // Configure grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (H + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (F + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Allocate device memory
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, F * G * sizeof(int));
    cudaMalloc(&d_B, G * H * sizeof(int));
    cudaMalloc(&d_C, F * H * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A.data(), F * G * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), G * H * sizeof(int), cudaMemcpyHostToDevice);

    // Copy data to constant memory
    cudaMemcpyToSymbol(const_A, h_A.data(), F * G * sizeof(int));
    cudaMemcpyToSymbol(const_B, h_B.data(), G * H * sizeof(int));

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

    // Launch the kernel
    matrixMulConstMemoryKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, F, G, H);

    // Measure kernel execution time
    cudaDeviceSynchronize();
    cudaEventRecord(stop_cuda_kernel);
    cudaEventSynchronize(stop_cuda_kernel);
    float ms_kernel;
    cudaEventElapsedTime(&ms_kernel, start_cuda_kernel, stop_cuda_kernel);

    // Copy the result back to the host
    cudaMemcpy(h_C.data(), d_C, F * H * sizeof(int), cudaMemcpyDeviceToHost);

    // Measure total time including data transfer
    cudaDeviceSynchronize();
    cudaEventRecord(stop_cuda_total);
    cudaEventSynchronize(stop_cuda_total);
    float ms_total;
    cudaEventElapsedTime(&ms_total, start_cuda_total, stop_cuda_total);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    info info;
    info.kernelTime = ms_kernel / 1000.0; // Convert to seconds
    info.totalTime = ms_total / 1000.0;   // Convert to seconds
    return info;
}




float getMedian(std::vector<float> &v)
{
    size_t n = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + n, v.end());
    return v[n];
}

int main(void)
{
    //make bigger and bigger matrix multiplication and get the median of the kernel time and total time
    //print # elements, kernel time, total time
    
    const int MAX_SIZE = 2500;
    const int numberOfReruns = 10;
    std::vector<std::vector<float>> totalTimesMatrix_global;
    std::vector<std::vector<float>> kernelTimesMatrix_global;
    std::vector<std::vector<float>> totalTimesMatrix_shared;
    std::vector<std::vector<float>> kernelTimesMatrix_shared;
    std::vector<std::vector<float>> totalTimesMatrix_const;
    std::vector<std::vector<float>> kernelTimesMatrix_const;

    for (int i = 8; i <= MAX_SIZE; i *= 2)
    {
        std::vector<float> totalTimes_global;
        std::vector<float> kernelTimes_global;
        std::vector<float> totalTimes_shared;
        std::vector<float> kernelTimes_shared;
        std::vector<float> totalTimes_const;
        std::vector<float> kernelTimes_const;

        for (int j = 0; j < numberOfReruns; j++)
        {
            info info_global = matrix_multiplication_run(i, i, i, i);
            info info_shared = matrix_multiplication_run_shared(i, i, i, i);
            info info_const = matrix_multiplication_run_const(i, i, i, i);

            totalTimes_global.push_back(info_global.totalTime);
            kernelTimes_global.push_back(info_global.kernelTime);
            totalTimes_shared.push_back(info_shared.totalTime);
            kernelTimes_shared.push_back(info_shared.kernelTime);
            totalTimes_const.push_back(info_const.totalTime);
            kernelTimes_const.push_back(info_const.kernelTime);
        }

        totalTimesMatrix_global.push_back(totalTimes_global);
        kernelTimesMatrix_global.push_back(kernelTimes_global);
        totalTimesMatrix_shared.push_back(totalTimes_shared);
        kernelTimesMatrix_shared.push_back(kernelTimes_shared);
        totalTimesMatrix_const.push_back(totalTimes_const);
        kernelTimesMatrix_const.push_back(kernelTimes_const);
    }

    for (int i = 0; i < totalTimesMatrix_global.size(); i++)
    {
        printf("%d, %.6e, %.6e, %.6e\n", (int)pow(2, 3 + i), 
        getMedian(kernelTimesMatrix_global[i]),
        getMedian(kernelTimesMatrix_shared[i]),
        getMedian(kernelTimesMatrix_const[i]));        
    }
    

    return 0;
}