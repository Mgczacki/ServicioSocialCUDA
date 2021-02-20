//Derivado de https://cuda-tutorial.readthedocs.io/
#define N 100000000
#include <iostream>
#include <math.h>

__global__ void vector_add(float *out, float *a, float *b, int n) {
    int indice = blockIdx.x * blockDim.x + threadIdx.x;//Indice del thread que ejecuta el kernel
    int paso = blockDim.x * gridDim.x;//El numero de threads por bloque
    for(int i = indice; i < n; i+=paso){
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
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);
    // Inicializamos a y b
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; b[i] = 2.0f;
    }
    //Copiamos los vectores a y b al GPU.
    cudaMemcpy(cuda_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);
    // Llamamos al kernel de CUDA (1 bloque, 256 threads por bloque).
    vector_add<<<100,256>>>(cuda_out, cuda_a, cuda_b, N);
    //Copiamos el vector de salida del GPU al anfitrión.
    cudaMemcpy(out, cuda_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(out[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;
    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_out);
    free(a);
    free(b);
    free(out);
    //Sugiero ver el comando: nvprof ./vector_add_parallel_multiblock
}
