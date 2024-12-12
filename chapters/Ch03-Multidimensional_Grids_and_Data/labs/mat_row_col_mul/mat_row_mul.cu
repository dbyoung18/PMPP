/*/
 * 
 *  A program that produce one output matrix row of matrix multiplication
 * 
 *  Compile with:
 *      nvcc mat_row_mul.cu
 * 
 *  Run with:
 *     ./a.out <M> <N> <K>
 * 
/*/


#include <stdio.h>
#include <cuda.h>


__global__ void mat_row_mul(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    for (int col = 0; col < N; ++col) {
        float sum = 0.f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}


int main(int argc, char** argv) {
    int M, N, K;

    float *h_A, h_B, h_C;
    float *d_A, d_B, d_C;

    if (argc != 4) {
        printf("Usage: ./a.out <M> <N> <K>\n");
        return 1;
    }

    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);

    // Allocate memory on host
    h_A = (float*)malloc(M * K * sizeof(float));
    h_B = (float*)malloc(K * N * sizeof(float));
    h_C = (float*)malloc(M * N * sizeof(float));

    // Initialize host memory
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < K; ++j)
            h_A[i * K + j] = 1;

    for (int i = 0; i < K; ++i)
        for (int j = 0; j < N; ++j)
            h_B[i * N + j] = 1;

    // Allocate memory on device
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));

    // Copy host memory to device memory
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 blockDim(N, 1, 1);
    dim3 gridDim(M, 1, 1);
    mat_row_mul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    // Copy device memory to host memory
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

#ifdef DEBUG
    // Print the result
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%f ", h_C[i][j]);
        }
        printf("\n");
    }
#endif

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    cudaFree(h_A);
    cudaFree(h_B);
    cudaFree(h_C);

    return 0;
}