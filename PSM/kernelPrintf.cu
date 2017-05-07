#include "kernelPrintf.h"


//__device__ void __syncthreads(void);
__global__ void kernelPrintf(int *O,int sizeO){
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i<sizeO){			
		printf("[%d]:%d ; ",i,O[i]);
	}

}



__global__ void kernelPrintFloat(float* A,int n){
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	if (i<n){
		printf("[%d]:%.0f ;",i,A[i]);
	}

}