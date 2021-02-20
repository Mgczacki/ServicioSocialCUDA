#include <iostream>
#include <math.h>
#define N 900
#define dimM1x 30
#define dimM1y 30
#define dimM2x 30
#define dimM2y 30

__global__ void matrix_multiply(float *out, float *a, float *b) {
    int indiceX = threadIdx.x;
    int indiceY = threadIdx.y;
    float accumulator = 0.0f;
    for(int i = 0; i < dimM1x; i++){//Calculamos el producto punto del vector renglon de M1 y vector columna de M2
        accumulator += a[(dimM1x*indiceX)+i]*b[(dimM2y*i)+indiceY];
    }
    out[(dimM2x*indiceY)+indiceX] = accumulator;
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
        b[i] = 1.0f;
    }
    // Llamamos al kernel de CUDA (1 bloque, 30x30 threads por bloque).
    matrix_multiply<<<1,dim3(30,30)>>>(out, a, b);
    cudaDeviceSynchronize();//Esperamos a que acaben todos los kernels.
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(out[i]-dimM1x));
    std::cout << "Max error: " << maxError << std::endl;
    cudaFree(a);
    cudaFree(b);
    cudaFree(out);
    //Sugiero ver el comando: nvprof ./simple_matrix_multiplication
}
