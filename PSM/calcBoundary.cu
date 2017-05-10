#include "calcBoundary.h"

__global__ void kernelCalcBoundary(Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,int *d_B,unsigned int Lv){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i<noElem_d_ValidExtension-1){
		unsigned int graphIdAfter=d_ValidExtension[i+1].vgi/Lv;
		unsigned int graphIdCurrent=d_ValidExtension[i].vgi/Lv;
		unsigned int resultDiff=graphIdAfter-graphIdCurrent;
		d_B[i]=resultDiff;
	}

}


cudaError_t calcBoundary(Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,int *d_B,unsigned int Lv){
	cudaError_t cudaStatus;

	dim3 block(1024);
	dim3 grid((noElem_d_ValidExtension+block.x)/block.x);

	kernelCalcBoundary<<<grid,block>>>(d_ValidExtension,noElem_d_ValidExtension,d_B,Lv);

	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaDeviceSynchronize calcBoundary failed");
		goto Error;
	}

Error:
	return cudaStatus;
}
