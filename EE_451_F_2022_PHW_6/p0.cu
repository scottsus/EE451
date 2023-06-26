#include <stdio.h>

#define N 512

inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}

__global__ void matrixMult(int *A, int *B, int *C)
{
    printf("Sequential Matrix Mult\n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

void log(int *A)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%d ", A[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char **argv)
{
    const int bytes = N * N * sizeof(int);

    int deviceId = 0;
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, deviceId));
    printf("Device: %s\n", prop.name);
    checkCuda(cudaSetDevice(deviceId));

    int *A, *B, *C, *d_A, *d_B, *d_C;
    checkCuda(cudaMallocHost((void **)&A, bytes));
    checkCuda(cudaMallocHost((void **)&B, bytes));
    checkCuda(cudaMallocHost((void **)&C, bytes));
    checkCuda(cudaMalloc((void **)&d_A, bytes));
    checkCuda(cudaMalloc((void **)&d_B, bytes));
    checkCuda(cudaMalloc((void **)&d_C, bytes));
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = i;
            B[i * N + j] = j;
            C[i * N + j] = 0;
        }
    }

    float ms;
    cudaEvent_t startEvent, stopEvent, dummyEvent;
    checkCuda(cudaEventCreate(&startEvent));
    checkCuda(cudaEventCreate(&stopEvent));
    checkCuda(cudaEventCreate(&dummyEvent));

    checkCuda(cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaEventRecord(startEvent, 0));
    checkCuda(cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_C, C, bytes, cudaMemcpyHostToDevice));

    dim3 dimGrid(64, 64);
    dim3 dimBlock(16, 16);

    matrixMult<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    checkCuda(cudaMemcpy(A, d_A, bytes, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost));
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));

    printf("Serial time (ms): %f\n", ms);
    printf("C[451][451]: %d\n", C[451 * N + 451]);

    checkCuda(cudaEventDestroy(startEvent));
    checkCuda(cudaEventDestroy(stopEvent));
    checkCuda(cudaEventDestroy(dummyEvent));
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);

    printf("Program exiting\n");
    return 0;
}
