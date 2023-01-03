#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Matrix size
#define N 2500

// Thread block size
#define BLOCK_SIZE 32

__global__ void MatrixMulKernel(float *a, float *b, float *c)
{
    // 2D Thread ID
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // Output element
    float Pvalue = 0;

    // Loop over all A and B elements
    for (int k = 0; k < N; ++k)
    {
        // Multiply-add
        Pvalue += a[tx * N + k] * b[k * N + ty];

    }

    // Save result
    c[ty * N + tx] = Pvalue;
}

int main()
{
    // Allocate host memory
    float *h_a = (float*)malloc(N * N * sizeof(float));
    float *h_b = (float*)malloc(N * N * sizeof(float));
    float *h_c = (float*)malloc(N * N * sizeof(float));

    // Initialize host memory
    for (int i = 0; i < N * N; ++i)
    {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
        h_c[i] = 0;
    }

    // Allocate device memory
    float *d_a;
    float *d_b;
    float *d_c;
    cudaMalloc((void**)&d_a, N * N * sizeof(float));
    cudaMalloc((void**)&d_b, N * N * sizeof(float));
    cudaMalloc((void**)&d_c, N * N * sizeof(float));

    // Copy host memory to device
    cudaMemcpy(d_a, h_a, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Setup execution configuration
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Start timer
    clock_t start = clock();

    // Launch kernel on the device
    MatrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);

    // Copy result from device to host
    cudaMemcpy(h_c, d_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Stop timer
    clock_t end = clock();

    // Print time
    printf("GPU computation time: %f s\n", (double)(end - start) / CLOCKS_PER_SEC);
    
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Start timer
    start = clock();

    // CPU matrix multiplication
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                h_c[i * N + j] += h_a[i * N + k] * h_b[k * N + j];

    // Stop timer
    end = clock();

    // Print time
    printf("CPU computation time: %f s\n", (double)(end - start) / CLOCKS_PER_SEC);

    // Free host memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
// Shutdown CUDA
    cudaDeviceReset();

    return 0;
}
