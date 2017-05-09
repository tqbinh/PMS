#include "calcLabelAndStoreUniqueExtension.h"

__global__ void kernelCalcLabelAndStoreUniqueExtension(int *d_allPossibleExtension,int *d_allPossibleExtensionScanResult,unsigned int noElem_allPossibleExtension,Extension *d_UniqueExtension,unsigned int Le,unsigned int Lv){
	int i=blockIdx.x*blockDim.x + threadIdx.x;	
	if(i<noElem_allPossibleExtension && d_allPossibleExtension[i]==1){
		int li,lj,lij;
		li=i/(Le*Lv);
		lij=(i%(Le*Lv))/Lv;
		lj=(i%(Le*Lv))-((i%(Le*Lv))/Lv)*Lv;
		//printf("\n[%d]:%d li:%d lij:%d lj:%d",i,d_allPossibleExtensionScanResult[i],li,lij,lj);
		d_UniqueExtension[d_allPossibleExtensionScanResult[i]].li=li;
		d_UniqueExtension[d_allPossibleExtensionScanResult[i]].lij=lij;
		d_UniqueExtension[d_allPossibleExtensionScanResult[i]].lj=lj;
	}
}

cudaError_t calcLabelAndStoreUniqueExtension(int *d_allPossibleExtension,int *d_allPossibleExtensionScanResult,unsigned int noElem_allPossibleExtension,Extension *d_UniqueExtension,unsigned int noElem_d_UniqueExtension,unsigned int Le,unsigned int Lv){
	cudaError_t cudaStatus;


	dim3 block(1024);
	dim3 grid((noElem_allPossibleExtension+block.x-1)/block.x);

	kernelCalcLabelAndStoreUniqueExtension<<<grid,block>>>(d_allPossibleExtension,d_allPossibleExtensionScanResult,noElem_allPossibleExtension,d_UniqueExtension,Le,Lv);

	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaDeviceSynchronize failed");
		goto Error;
	}

Error:
	return cudaStatus;
}
