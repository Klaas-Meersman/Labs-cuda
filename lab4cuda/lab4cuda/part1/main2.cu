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

// #define M 512       // Lenna width
// #define N 512       // Lenna height

#define C 3       // Colors
#define OFFSET 16 // Header length


uint8_t *create_random_image_array(int width, int height, int offset) {
    // Seed the random number generator
    srand(time(NULL));

    // Allocate memory for the image array including the offset
    uint8_t *image_array = (uint8_t *)malloc(width * height * 3 * sizeof(uint8_t) + offset);
    if (image_array == NULL) {
        perror("ERROR: Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    // Fill the array with random values (0-255) for RGB components
    for (int i = offset; i < width * height * 3 + offset; i++) {
        image_array[i] = rand() % 256; // Random value between 0 and 255
    }

    // Return the array starting from the offset
    return image_array + offset;
}

void save_image_array(uint8_t *image_array,int M, int N){
    /*
     * Save the data of an (RGB) image as a pixel map.
     *
     * Parameters:
     *  - param1: The data of an (RGB) image as a 1D array
     *
     */
    // Try opening the file
    FILE *imageFile;
    imageFile = fopen("./output_image.ppm", "wb");
    if (imageFile == NULL)
    {
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }

    // Configure the file
    fprintf(imageFile, "P6\n");          // P6 filetype
    fprintf(imageFile, "%d %d\n", M, N); // dimensions
    fprintf(imageFile, "255\n");         // Max pixel

    // Write the image
    fwrite(image_array, 1, M * N * C, imageFile);

    // Close the file
    fclose(imageFile);
}



struct info{
    float kernelTime;
    float totalTime;
};

__global__ void REDfilterGPU_Uncoalesced(uint8_t *original, uint8_t *inverted, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size){
        if (i % 3 == 0){
            inverted[i] = 255 - original[i];
        }
    }
}


info REDfilterGPUrun_Uncoalesced(uint8_t *image_array, int M, int N){ 
    // Read the image
    //uint8_t *image_array = get_image_array();
    // Allocate host output
    uint8_t *new_image_array = (uint8_t *)malloc(M * N * C);

    // Set up grid and block dimensions
    int blockSize = 256;
    int numBlocks = (M * N * C + blockSize - 1) / blockSize;

    // Allocate device memory
    uint8_t *d_original, *d_inverted;
    cudaMalloc((void **)&d_original, M * N * C * sizeof(uint8_t));
    cudaMalloc((void **)&d_inverted, M * N * C * sizeof(uint8_t));

    // Measure total time including data transfer
    cudaEvent_t start_cuda_total, stop_cuda_total;
    cudaEventCreate(&start_cuda_total);
    cudaEventCreate(&stop_cuda_total);
    cudaDeviceSynchronize();
    cudaEventRecord(start_cuda_total);

    // Copy data to device
    cudaMemcpy(d_original, image_array, M * N * C * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Measure kernel execution time separately
    cudaEvent_t start_cuda_kernel, stop_cuda_kernel;
    cudaEventCreate(&start_cuda_kernel);
    cudaEventCreate(&stop_cuda_kernel);
    cudaDeviceSynchronize();
    cudaEventRecord(start_cuda_kernel);

    // Launch kernel
    REDfilterGPU_Uncoalesced<<<numBlocks, blockSize>>>(d_original, d_inverted, M * N * C);

    // Measure kernel execution time
    cudaDeviceSynchronize();
    cudaEventRecord(stop_cuda_kernel);
    cudaEventSynchronize(stop_cuda_kernel);
    float ms_kernel;
    cudaEventElapsedTime(&ms_kernel, start_cuda_kernel, stop_cuda_kernel);

    // Copy result back to host
    cudaMemcpy(new_image_array, d_inverted, M * N * C * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Measure total time including data transfer
    cudaDeviceSynchronize();
    cudaEventRecord(stop_cuda_total);
    cudaEventSynchronize(stop_cuda_total);
    float ms_total;
    cudaEventElapsedTime(&ms_total, start_cuda_total, stop_cuda_total);

    // Save the image
    save_image_array(new_image_array,M,N);

    // Free memory
    free(new_image_array);
    cudaFree(d_original);
    cudaFree(d_inverted);

    info info;
    info.kernelTime = ms_kernel / 1000.0; // Convert to seconds
    info.totalTime = ms_total / 1000.0;   // Convert to seconds
    return info;
}



void transformImageData(uint8_t *original, uint8_t *transformed, int size){
    for (int i = 0; i < size / 3; ++i){
        // Place red pixels first
        transformed[i] = original[i * 3]; // Red channel

        // Then green pixels
        transformed[size / 3 + i] = original[i * 3 + 1]; // Green channel

        // Then blue pixels
        transformed[2 * size / 3 + i] = original[i * 3 + 2]; // Blue channel
    }
}

__global__ void redChannelFilter(uint8_t* transformed, uint8_t* inverted, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size / 3) {
        int redIndex = i; // Red pixels are in the first third of the array

        inverted[redIndex] = 255 - transformed[redIndex]; // Inversion of blue channel
    }
}


void reverseTransformImageData(uint8_t *inverted, uint8_t *finalImage, int size){
    for (int i = 0; i < size / 3; ++i){
        // Place the red pixels back to their original location
        finalImage[i * 3] = inverted[i];

        // Place the green pixels back to their original location
        finalImage[i * 3 + 1] = inverted[size / 3 + i];

        // Place the blue pixels back to their original location
        finalImage[i * 3 + 2] = inverted[2 * size / 3 + i];
    }
}

info REDfilterGPUrun_Coalesced(uint8_t *image_array, int M, int N) {
    // Read the image
    //uint8_t *image_array = get_image_array();
    
    
    // Allocate host output and transformed input
    uint8_t *new_image_array = (uint8_t *)malloc(M * N * C * sizeof(uint8_t));
    uint8_t *transformed_input = (uint8_t *)malloc(M * N * C * sizeof(uint8_t));

    // Transform the input data
    transformImageData(image_array, transformed_input, M * N * C);

    // Set up grid and block dimensions
    int blockSize = 256;
    int numBlocks = (M * N * C + blockSize - 1) / blockSize;

    // Allocate device memory
    uint8_t *d_transformed, *d_inverted;
    cudaMalloc((void **)&d_transformed, M * N * C * sizeof(uint8_t));
    cudaMalloc((void **)&d_inverted, M * N * C * sizeof(uint8_t));

    // Measure total time including data transfer
    cudaEvent_t start_cuda_total, stop_cuda_total;
    cudaEventCreate(&start_cuda_total);
    cudaEventCreate(&stop_cuda_total);
    cudaDeviceSynchronize();
    cudaEventRecord(start_cuda_total);

    // Copy transformed data to device
    cudaMemcpy(d_transformed, transformed_input, M * N * C * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Measure kernel execution time separately
    cudaEvent_t start_cuda_kernel, stop_cuda_kernel;
    cudaEventCreate(&start_cuda_kernel);
    cudaEventCreate(&stop_cuda_kernel);
    cudaDeviceSynchronize();
    cudaEventRecord(start_cuda_kernel);

    // Launch kernels
    redChannelFilter<<<numBlocks, blockSize>>>(d_transformed, d_inverted, M * N * C);
    //greenChannelFilter<<<numBlocks, blockSize>>>(d_transformed, d_inverted, M * N * C);
    //blueChannelFilter<<<numBlocks, blockSize>>>(d_transformed, d_inverted, M * N * C);

    // Measure kernel execution time
    cudaDeviceSynchronize();
    cudaEventRecord(stop_cuda_kernel);
    cudaEventSynchronize(stop_cuda_kernel);
    float ms_kernel;
    cudaEventElapsedTime(&ms_kernel, start_cuda_kernel, stop_cuda_kernel);

    // Copy the result back to the host
    cudaMemcpy(transformed_input, d_inverted, M * N * C * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Reverse transform the data
    reverseTransformImageData(transformed_input, new_image_array, M * N * C);

    // Measure total time including data transfer
    cudaDeviceSynchronize();
    cudaEventRecord(stop_cuda_total);
    cudaEventSynchronize(stop_cuda_total);
    float ms_total;
    cudaEventElapsedTime(&ms_total, start_cuda_total, stop_cuda_total);

    // Save the image
    save_image_array(new_image_array,M,N);

    // Free memory
    free(new_image_array);
    free(transformed_input);
    cudaFree(d_transformed);
    cudaFree(d_inverted);

    info info;
    info.kernelTime = ms_kernel / 1000.0; // Convert to seconds
    info.totalTime = ms_total / 1000.0;   // Convert to seconds
    return info;
}


float getMedian(std::vector<float> &v){
    size_t n = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + n, v.end());
    return v[n];
}


int main(void){
    const int NUM_RUNS = 20;
    const int NUM_IMAGE_SIZES = 6;

    // Create vectors to store the times for uncoalesced and coalesced runs for each image size
    std::vector<std::vector<float>> uncoalesced_times(NUM_IMAGE_SIZES, std::vector<float>(NUM_RUNS));
    std::vector<std::vector<float>> coalesced_times(NUM_IMAGE_SIZES, std::vector<float>(NUM_RUNS));

    // Array of different image heights and widths
    int image_dimensions[NUM_IMAGE_SIZES][2] = {
        {512, 512},    // 512x512
        {960, 1280},   // 960x1280
        {1920, 1080},  // 1920x1080
        {3840, 2160},   // 3840x2160
        {4096, 4096},   // 4096x2160
        {7680, 4320}   // 7680x4320
    };

    // For all these sizes, generate a random image and run the filters
    for (int i = 0; i < NUM_IMAGE_SIZES; i++) {
        int M = image_dimensions[i][0];
        int N = image_dimensions[i][1];
        uint8_t *random_image_array = create_random_image_array(M, N, OFFSET);

        // Warm-up run
        REDfilterGPUrun_Uncoalesced(random_image_array, M, N);
        REDfilterGPUrun_Coalesced(random_image_array, M, N);

        // Run the filters NUM_RUNS times and store the runtimes
        for (int run = 0; run < NUM_RUNS; run++) {
            uncoalesced_times[i][run] = REDfilterGPUrun_Uncoalesced(random_image_array, M, N).kernelTime;
            coalesced_times[i][run] = REDfilterGPUrun_Coalesced(random_image_array, M, N).kernelTime;
        }
    }

    // Print the results in a table-like format
    for (int i = 0; i < NUM_IMAGE_SIZES; i++) {
        float median_uncoalesced = getMedian(uncoalesced_times[i]);
        float median_coalesced = getMedian(coalesced_times[i]);

        printf("%d,%.6e,%.6e\n", image_dimensions[i][0] * image_dimensions[i][1], median_uncoalesced, median_coalesced);
    }

    return 0;
}