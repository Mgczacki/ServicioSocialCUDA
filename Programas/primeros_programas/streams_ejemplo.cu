//Inspirado en https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/overlap-data-transfers/async.cu
//https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/
#define N 400000000
#include <iostream>
#include <math.h>

__global__ void vector_add(float *out, float *a, float *b, int n_max, int offset) {
    int indice = offset + blockIdx.x * blockDim.x + threadIdx.x;//Indice del thread que ejecuta el kernel
    int paso = blockDim.x * gridDim.x;//El numero de threads por bloque
    for(int i = indice; i < n_max; i+=paso){
        out[i] = a[i] + b[i];
    }
}

int main(){
    float *a, *b, *out; //Apuntadores a memoria del anfitrión
    float *cuda_a, *cuda_b, *cuda_out; //Apuntadores a memoria del GPU
    //Generamos los arreglos en memoria del GPU
    cudaMalloc((void**)&cuda_a, sizeof(float) * N);
    cudaMalloc((void**)&cuda_b, sizeof(float) * N);
    cudaMalloc((void**)&cuda_out, sizeof(float) * N);
    //Generamos los arreglos en memoria del anfitrión
    cudaMallocHost((void**)&a, sizeof(float) * N);
    cudaMallocHost((void**)&b, sizeof(float) * N);
    cudaMallocHost((void**)&out, sizeof(float) * N);
    //a   = (float*)malloc(sizeof(float) * N);
    //b   = (float*)malloc(sizeof(float) * N);
    //out = (float*)malloc(sizeof(float) * N);
    // Inicializamos a y b
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; b[i] = 2.0f;
    }
    
    float ms;
    cudaEvent_t startEvent, stopEvent, dummyEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventCreate(&dummyEvent);
    //Versión con un solo stream.
    cudaEventRecord(startEvent,0);
    cudaMemcpy(cuda_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);
    vector_add<<<100, 256>>>(cuda_out, cuda_a, cuda_b, N, 0);
    cudaMemcpy(out, cuda_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    printf("Tiempo para un solo stream (ms): %f\n", ms);
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(out[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;
    //Versiones con múltiples streams - Iteraciones múltiples
    //Haremos 50 streams distintos.
    int n_streams = 50;
    cudaStream_t stream[n_streams];
    for (int i = 0; i < n_streams; i ++)
    {
        cudaStreamCreate(&stream[i]);
    }
    cudaEventRecord(startEvent,0);
    int streamSize = N / n_streams;
    int streamBytes = sizeof(float) * N / n_streams;
    for (int i = 0; i < n_streams; i ++) 
    {
        int offset = i * streamSize;
        cudaMemcpyAsync(&cuda_a[offset], &a[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(&cuda_b[offset], &b[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
    }
    for (int i = 0; i < n_streams; ++i)
    {
        int offset = i * streamSize;
        vector_add<<<100, 256, 0, stream[i]>>>(cuda_out, cuda_a, cuda_b, offset+streamSize, offset);
    }
    for (int i = 0; i < n_streams; i ++) 
    {
        int offset = i * streamSize;
        cudaMemcpyAsync(&out[offset], &cuda_out[offset], streamBytes, cudaMemcpyDeviceToHost, stream[i]);
    }
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    printf("Tiempo con múltiples streams - Separación de tareas(ms): %f\n", ms);
    maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(out[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;
    //Versión con múltiples streams - Una iteración por stream.
    cudaEventRecord(startEvent,0);
    for (int i = 0; i < n_streams; i ++) 
    {
        int offset = i * streamSize;
        cudaMemcpyAsync(&cuda_a[offset], &a[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(&cuda_b[offset], &b[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
        vector_add<<<100, 256, 0, stream[i]>>>(cuda_out, cuda_a, cuda_b, offset+streamSize, offset);
        cudaMemcpyAsync(&out[offset], &cuda_out[offset], streamBytes, cudaMemcpyDeviceToHost, stream[i]);
    }
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    printf("Tiempo con múltiples streams - Unificación de tareas (ms): %f\n", ms);
    maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(out[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;
    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_out);
    cudaFree(a);
    cudaFree(b);
    cudaFree(out);
    //Sugiero ver el comando: nvprof ./vector_add_parallel_multiblock
}
