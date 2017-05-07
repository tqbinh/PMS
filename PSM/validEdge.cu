#include "validEdge.h"
#include "kernelPrintf.h"

__global__ void	kernelValidEdge(Extension *d_Extension,int *V,unsigned int numberElementd_Extension){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i<numberElementd_Extension){	
		if(d_Extension[i].li<=d_Extension[i].lj){
			V[i]=1;
		}
	}
}




cudaError_t validEdge(Extension *d_Extension,int *V,unsigned int numberElementd_Extension){
	cudaError_t cudaStatus;

	dim3 block(512);
	dim3 grid(numberElementd_Extension+block.x-1/block.x);

	kernelValidEdge<<<grid,block>>>(d_Extension,V,numberElementd_Extension);
	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaDeviceSynchronize kernelValidEdge failed",cudaStatus);
		goto labelError;
	}
	//
	printf("\nV array: ");
	kernelPrintf<<<grid,block>>>(V,numberElementd_Extension);

labelError:

	return cudaStatus;
}
