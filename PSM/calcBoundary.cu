#include "calcBoundary.h"

__global__ void kernelCalcBoundary(Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,int *d_B,unsigned int maxOfVer){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i<noElem_d_ValidExtension-1){
		unsigned int graphIdAfter=d_ValidExtension[i+1].vgi/maxOfVer;
		unsigned int graphIdCurrent=d_ValidExtension[i].vgi/maxOfVer;
		unsigned int resultDiff=graphIdAfter-graphIdCurrent;
		d_B[i]=resultDiff;
	}

}


cudaError_t calcBoundary(Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,int *d_B,unsigned int maxOfVer){
	cudaError_t cudaStatus;

	dim3 block(1024);
	dim3 grid((noElem_d_ValidExtension+block.x)/block.x);

	kernelCalcBoundary<<<grid,block>>>(d_ValidExtension,noElem_d_ValidExtension,d_B,maxOfVer);

	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaDeviceSynchronize calcBoundary failed");
		goto Error;
	}

Error:
	return cudaStatus;
}
