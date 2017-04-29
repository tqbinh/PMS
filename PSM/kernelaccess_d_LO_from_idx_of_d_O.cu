#include "kernelaccess_d_LO_from_idx_of_d_O.h"
__device__ void __syncthreads(void);

inline __global__ void kernelaccess_d_LO_from_idx_of_d_O(int *d_LO,int *d_N,int sizeOfArrayN){
	int i=threadIdx.x + blockDim.x*blockIdx.x;
	if(i<sizeOfArrayN){		
			printf("d_n[%d]:%d ",d_N[i],d_LO[d_N[i]]);		
	}
	__syncthreads();
	
}