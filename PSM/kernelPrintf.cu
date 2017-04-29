#include "kernelPrintf.h"


//__device__ void __syncthreads(void);
__global__ void kernelPrintf(int *O,int sizeO){
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i<sizeO){
		printf("[%d]:%d ; ",i,O[i]);
	}

}
