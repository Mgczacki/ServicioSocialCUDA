#include <iostream>

__global__ void identidad() {
    printf("Bloque: (%d,%d,%d) Hilo: (%d,%d,%d)\n",  
                blockIdx.x, blockIdx.y, blockIdx.z,  
                threadIdx.x, threadIdx.y,  threadIdx.z);
}

int main(){
    
    // Llamamos al kernel de CUDA (4 bloques, 2 threads por bloque).
    printf("====== Unidimensionales ======\n");
    identidad<<<4,2>>>();//Aqui se usan bloques y threads unidimensionales.
    cudaDeviceSynchronize();//Esperamos a que acaben todos los kernels.
    dim3 blockSize2d = dim3( 2, 3 );
    dim3 gridSize2d  = dim3( 3, 2 );
    printf("====== Bidimensionales ======\n");
    identidad<<<gridSize2d,blockSize2d>>>();//Ejemplo de bloques y threads bidimensionales
    cudaDeviceSynchronize();//Esperamos a que acaben todos los kernels.
    dim3 blockSize3d = dim3( 2, 3, 4);
    dim3 gridSize3d  = dim3( 4, 3, 2 );
    printf("====== Tridimensionales ======\n");
    identidad<<<gridSize3d,blockSize3d>>>();//Ejemplo de bloques y threads tridimensionales
    cudaDeviceSynchronize();//Esperamos a que acaben todos los kernels.

    //Todos los threads y bloques en CUDA utilizan índices tridimensionales.
}
