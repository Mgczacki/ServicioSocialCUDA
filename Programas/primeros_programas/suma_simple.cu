#include <iostream>
__global__ void add( int a, int b, int *c ) {//Función del kernel
 *c = a + b;
}
int main( void ) {
 int c;
 int *dev_c;
 cudaMalloc((void**)&dev_c, sizeof(int));
 add<<<1,1>>>( 3, 4, dev_c );//Se ejecuta el kernel en un bloque con un thread.
 cudaMemcpy( &c, dev_c, sizeof(int), cudaMemcpyDeviceToHost );
 printf( "3 + 4 = %d\n", c );
 cudaFree( dev_c );
 return 0;
}
