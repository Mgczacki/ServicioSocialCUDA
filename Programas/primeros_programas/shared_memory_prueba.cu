//Derivado de https://cuda-tutorial.readthedocs.io/ y
//https://developer.nvidia.com/blog/unified-memory-cuda-beginners/
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
    float *a, *b, *out; //Apuntadores a memoria compartida
    //Generamos los arreglos de memoria compartida
    cudaMallocManaged(&a, sizeof(float) * N);
    cudaMallocManaged(&b, sizeof(float) * N);
    cudaMallocManaged(&out, sizeof(float) * N);
    // Inicializamos a y b
    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
    // Llamamos al kernel de CUDA (1 bloque, 256 threads por bloque).
    vector_add<<<100,256>>>(out, a, b, N);
    cudaDeviceSynchronize();//Esperamos a que acaben todos los kernels.
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(out[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;
    cudaFree(a);
    cudaFree(b);
    cudaFree(out);
    //Sugiero ver el comando: nvprof ./shared_memory_prueba
}
